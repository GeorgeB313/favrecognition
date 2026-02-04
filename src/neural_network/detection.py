import io
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from src.neural_network.inference import ModelNotReadyError

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled at runtime
    YOLO = None
    _ULTRALYTICS_IMPORT_ERROR = exc
else:  # pragma: no cover
    _ULTRALYTICS_IMPORT_ERROR = None


class FruitVegDetector:
    def __init__(
        self,
        model_path: Union[str, Path] = "models/fruitveg_detector.pt",
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        # Încarcă YOLO doar dacă dependența există și checkpoint-ul e prezent.
        if YOLO is None:
            raise ModelNotReadyError(
                "Ultralytics is not installed. Add 'ultralytics' to requirements.txt and install it."
            ) from _ULTRALYTICS_IMPORT_ERROR
        if not self.model_path.exists():
            raise ModelNotReadyError(
                f"Detector checkpoint missing at {self.model_path}. Train/export a detection model first."
            )
        model = YOLO(str(self.model_path))
        return model

    def detect(
        self,
        image_bytes: bytes,
        conf: float = 0.35,
        iou: float = 0.45,
        max_det: int = 10,
    ) -> List[dict]:
        # Rulează detecția și întoarce box-urile sortate după confidence.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return []

        detections = []
        names = result.names
        xyxy = result.boxes.xyxy.cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        clss = result.boxes.cls.cpu().tolist()

        for (x1, y1, x2, y2), score, cls_idx in zip(xyxy, confs, clss):
            label = names.get(int(cls_idx), str(int(cls_idx))) if isinstance(names, dict) else names[int(cls_idx)]
            detections.append(
                {
                    "label": label,
                    "confidence": float(score),
                    "box": {
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(max(0.0, x2 - x1)),
                        "h": float(max(0.0, y2 - y1)),
                    },
                }
            )

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections
