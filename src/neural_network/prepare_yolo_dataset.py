import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits from a flat folder")
    parser.add_argument("--images-dir", type=str, default="data/yolo_raw/images")
    parser.add_argument("--labels-dir", type=str, default="data/yolo_raw/labels")
    parser.add_argument("--output-dir", type=str, default="data/yolo")
    parser.add_argument("--split", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exts", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"])
    return parser.parse_args()


def get_images(images_dir: Path, exts: List[str]) -> List[Path]:
    images = []
    for ext in exts:
        images.extend(images_dir.rglob(f"*{ext}"))
    return sorted(set(images))


def split_list(items: List[Path], split: Tuple[float, float, float]) -> Tuple[List[Path], List[Path], List[Path]]:
    train_ratio, val_ratio, test_ratio = split
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def copy_pairs(items: List[Path], images_dir: Path, labels_dir: Path, out_root: Path, split: str) -> int:
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in items:
        rel = img_path.relative_to(images_dir)
        label_path = labels_dir / rel.with_suffix(".txt")
        if not label_path.exists():
            continue
        shutil.copy2(img_path, out_images / rel.name)
        shutil.copy2(label_path, out_labels / rel.with_suffix(".txt").name)
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_root = Path(args.output_dir)

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    images = get_images(images_dir, args.exts)
    if not images:
        raise SystemExit("No images found.")

    random.seed(args.seed)
    random.shuffle(images)
    train_items, val_items, test_items = split_list(images, tuple(args.split))

    copied_train = copy_pairs(train_items, images_dir, labels_dir, out_root, "train")
    copied_val = copy_pairs(val_items, images_dir, labels_dir, out_root, "val")
    copied_test = copy_pairs(test_items, images_dir, labels_dir, out_root, "test")

    print(f"Copied train: {copied_train}, val: {copied_val}, test: {copied_test}")


if __name__ == "__main__":
    main()
