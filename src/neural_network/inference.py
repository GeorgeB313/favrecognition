import io
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.neural_network.model import build_backbone_model  # noqa: E402


class ModelNotReadyError(RuntimeError):
    """Raised when the trained weights or label map are missing."""


class FruitVegPredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/fruitveg_cnn.pt",
        label_map_path: Union[str, Path] = "models/label_map.json",
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._class_names = self._load_class_names()
        self.model, self.transform, self._use_tta = self._load_model()

    def _load_class_names(self) -> List[str]:
        # Încarcă mapping-ul de clase folosit la inferență.
        if not self.label_map_path.exists():
            raise ModelNotReadyError(
                f"Label map missing at {self.label_map_path}. Run the training script first."
            )
        with self.label_map_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _load_model(self) -> Tuple[torch.nn.Module, transforms.Compose, bool]:
        # Reface backbone-ul și atașează greutățile salvate.
        if not self.model_path.exists():
            raise ModelNotReadyError(
                f"Model checkpoint missing at {self.model_path}. Train the model before serving predictions."
            )
        checkpoint = torch.load(self.model_path, map_location=self.device)
        backbone = checkpoint.get("backbone", "custom")
        # Never download weights at inference time; the checkpoint has trained weights.
        model = build_backbone_model(num_classes=checkpoint["num_classes"], backbone=backbone, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        image_size = checkpoint.get("image_size", 224)
        mean = checkpoint.get("mean", [0.5, 0.5, 0.5])
        std = checkpoint.get("std", [0.5, 0.5, 0.5])

        if isinstance(mean, (int, float)):
            mean = [float(mean)] * 3
        if isinstance(std, (int, float)):
            std = [float(std)] * 3

        if backbone != "custom":
            resize_size = int(round(image_size * 256 / 224))
            transform = transforms.Compose(
                [
                    transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        use_tta = backbone != "custom"
        return model, transform, use_tta

    def _predict_probs(self, image: Image.Image) -> torch.Tensor:
        # Calculează probabilitățile (cu TTA dacă e activ).
        tensors = []
        if self._use_tta:
            tensors.append(self.transform(image))
            tensors.append(self.transform(ImageOps.mirror(image)))
        else:
            tensors.append(self.transform(image))

        batch = torch.stack(tensors, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            probs = probs.mean(dim=0)
        return probs

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[dict]:
        # Returnează top-k predicții pentru o imagine.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        probs = self._predict_probs(image)
        top_probs, top_indices = probs.topk(min(top_k, probs.numel()))

        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            results.append(
                {
                    "label": self._class_names[idx],
                    "probability": prob,
                }
            )
        return results

    def predict_with_rejection(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        min_confidence: float = 0.60,
        min_margin: float = 0.10,
        unknown_label: str = "necunoscut",
    ) -> Tuple[bool, List[dict]]:
        predictions = self.predict(image_bytes, top_k=top_k)
        if not predictions:
            return False, [{"label": unknown_label, "probability": 0.0}]

        top1 = predictions[0]
        top1_prob = float(top1.get("probability", 0.0))
        top2_prob = float(predictions[1].get("probability", 0.0)) if len(predictions) > 1 else 0.0
        is_accepted = (top1_prob >= float(min_confidence)) and ((top1_prob - top2_prob) >= float(min_margin))
        if is_accepted:
            return True, predictions

        # Return the normal top-k too (useful for debugging), but mark as not accepted.
        return False, predictions

    @property
    def class_names(self) -> List[str]:
        return self._class_names
