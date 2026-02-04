import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

# Allow running as a script without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.neural_network.model import build_backbone_model  # noqa: E402

CHANNEL_MEAN = [0.5, 0.5, 0.5]
CHANNEL_STD = [0.5, 0.5, 0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_TRANSLATIONS = {
    "apple": "mar",
    "banana": "banana",
    "beetroot": "sfecla rosie",
    "bell pepper": "ardei gras",
    "cabbage": "varza",
    "capsicum": "ardei capia",
    "carrot": "morcov",
    "cauliflower": "conopida",
    "chilli pepper": "ardei iute",
    "corn": "porumb",
    "cucumber": "castravete",
    "eggplant": "vanata",
    "garlic": "usturoi",
    "ginger": "ghimbir",
    "grapes": "struguri",
    "jalepeno": "jalapeno",
    "kiwi": "kiwi",
    "lemon": "lamaie",
    "lettuce": "salata verde",
    "mango": "mango",
    "onion": "ceapa",
    "orange": "portocala",
    "paprika": "ardei paprika",
    "pear": "para",
    "peas": "mazare",
    "pineapple": "ananas",
    "pomegranate": "rodie",
    "potato": "cartof",
    "raddish": "ridiche",
    "soy beans": "boabe de soia",
    "spinach": "spanac",
    "sweetcorn": "porumb dulce",
    "sweetpotato": "cartof dulce",
    "tomato": "rosie",
    "turnip": "nap",
    "watermelon": "pepene verde",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN to recognize fruits and vegetables from RGB images.",
    )
    parser.add_argument("--train-dir", default="data/train", help="Folder with training images (ImageFolder structure).")
    parser.add_argument(
        "--val-dir",
        default="data/validation",
        help="Folder with validation images (optional).",
    )
    parser.add_argument("--test-dir", default="data/test", help="Folder with test images (optional).")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer.")
    parser.add_argument("--image-size", type=int, default=224, help="Input resolution for the CNN.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Folder where the trained model and label map will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--backbone",
        default="custom",
        choices=["custom", "mobilenet_v3_small"],
        help="Model backbone. Use a torchvision backbone for better generalization.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="If set, initialize torchvision backbone with ImageNet weights (training only).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint to resume/fine-tune from (loads weights and metadata).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available (useful on machines without GPU drivers).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int,
    mean: List[float],
    std: List[float],
    train: bool,
    imagenet_style: bool,
) -> transforms.Compose:
    if not train:
        if imagenet_style:
            resize_size = int(round(image_size * 256 / 224))
            return transforms.Compose(
                [
                    transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # Train transforms (with augmentations). For pretrained backbones, use an ImageNet-like
    # input pipeline that preserves aspect ratio via RandomResizedCrop.
    if imagenet_style:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.6, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # Custom CNN pipeline (kept simple and deterministic-ish).
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_loader(
    directory: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> Optional[Tuple[datasets.ImageFolder, DataLoader]]:
    path = Path(directory)
    if not path.exists():
        return None
    dataset = datasets.ImageFolder(directory, transform=transform)
    if len(dataset) == 0:
        return None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss, running_correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def save_artifacts(
    model: nn.Module,
    class_names: List[str],
    checkpoint_path: Path,
    label_map_path: Path,
    image_size: int,
    mean: List[float],
    std: List[float],
    backbone: str,
    pretrained: bool,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(class_names),
            "image_size": image_size,
            "mean": mean,
            "std": std,
            "backbone": backbone,
            "pretrained": pretrained,
        },
        checkpoint_path,
    )
    with label_map_path.open("w", encoding="utf-8") as fp:
        json.dump(class_names, fp, ensure_ascii=False, indent=2)


def to_romanian_labels(class_names: List[str]) -> List[str]:
    translated = []
    for name in class_names:
        translated.append(LABEL_TRANSLATIONS.get(name, name))
    return translated


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    resume_ckpt = None
    backbone = args.backbone
    image_size = args.image_size
    mean = None
    std = None
    pretrained_flag = bool(args.pretrained)

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        backbone = resume_ckpt.get("backbone", backbone)
        image_size = int(resume_ckpt.get("image_size", image_size))
        mean = resume_ckpt.get("mean", None)
        std = resume_ckpt.get("std", None)
        pretrained_flag = bool(resume_ckpt.get("pretrained", pretrained_flag))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    if mean is None or std is None:
        if backbone == "custom":
            mean, std = CHANNEL_MEAN, CHANNEL_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD

    imagenet_style = backbone != "custom"

    train_transform = build_transforms(image_size, mean, std, train=True, imagenet_style=imagenet_style)
    eval_transform = build_transforms(image_size, mean, std, train=False, imagenet_style=imagenet_style)
    pin_memory = device.type == "cuda"
    train_bundle = make_loader(
        args.train_dir,
        train_transform,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
    )
    if train_bundle is None:
        raise RuntimeError(f"No training images found in {args.train_dir}.")
    train_dataset, train_loader = train_bundle

    val_bundle = make_loader(
        args.val_dir,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_bundle = make_loader(
        args.test_dir,
        eval_transform,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )

    class_names_en = train_dataset.classes
    class_names_ro = to_romanian_labels(class_names_en)

    # When resuming, never download pretrained weights again.
    model = build_backbone_model(
        num_classes=len(class_names_en),
        backbone=backbone,
        pretrained=(pretrained_flag if resume_ckpt is None else False),
    ).to(device)

    if resume_ckpt is not None:
        if int(resume_ckpt.get("num_classes", len(class_names_en))) != len(class_names_en):
            raise RuntimeError(
                "Resume checkpoint num_classes does not match current dataset classes. "
                f"ckpt={resume_ckpt.get('num_classes')} dataset={len(class_names_en)}"
            )
        model.load_state_dict(resume_ckpt["model_state_dict"])
        print(f"Resumed weights from: {args.resume_from}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = (0.0, 0.0)
        if val_bundle is not None:
            _, val_loader = val_bundle
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

        if val_bundle is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
        elif val_bundle is None and (best_state is None or train_acc > best_val_acc):
            best_val_acc = train_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    if test_bundle is not None:
        _, test_loader = test_bundle
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
        print(f"Test set -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "fruitveg_cnn.pt"
    label_map_path = output_dir / "label_map.json"
    save_artifacts(
        model,
        class_names_ro,
        checkpoint_path,
        label_map_path,
        image_size,
        mean,
        std,
        backbone,
        bool(pretrained_flag),
    )

    print(f"Model saved to {checkpoint_path}")
    print(f"Label map saved to {label_map_path}")
    print("Clasa (EN) -> Eticheta (RO):")
    for en, ro in zip(class_names_en, class_names_ro):
        print(f" - {en} -> {ro}")


if __name__ == "__main__":
    main()
