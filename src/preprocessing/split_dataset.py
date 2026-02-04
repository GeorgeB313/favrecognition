import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split processed images into train/validation/test folders.",
    )
    parser.add_argument("--source", default="data/processed", help="Directory with processed images grouped by class.")
    parser.add_argument("--train", default="data/train", help="Output directory for training set.")
    parser.add_argument("--val", default="data/validation", help="Output directory for validation set.")
    parser.add_argument("--test", default="data/test", help="Output directory for test set.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of images per class for validation.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of images per class for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving (default moves to reduce disk usage).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing contents of destination folders before splitting.",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                item.unlink()
        for subdir in sorted([p for p in path.iterdir() if p.is_dir()], reverse=True):
            shutil.rmtree(subdir)
    path.mkdir(parents=True, exist_ok=True)


def collect_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()]


def dispatch_files(pairs: List[Tuple[Path, Path]], copy: bool) -> None:
    operation = shutil.copy2 if copy else shutil.move
    for src, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        operation(src, dest)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {source_dir}")

    train_dir = Path(args.train)
    val_dir = Path(args.val)
    test_dir = Path(args.test)

    if args.clean:
        for target in (train_dir, val_dir, test_dir):
            ensure_empty_dir(target)
    else:
        for target in (train_dir, val_dir, test_dir):
            target.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
        images = collect_images(class_dir)
        if not images:
            continue
        random.shuffle(images)

        val_count = int(len(images) * args.val_ratio)
        test_count = int(len(images) * args.test_ratio)
        train_count = len(images) - val_count - test_count

        # Ensure at least one sample goes to train when possible
        if train_count <= 0:
            train_count = max(1, len(images) - val_count - test_count)
            val_count = max(0, len(images) - train_count - test_count)

        class_name = class_dir.name
        train_dest = train_dir / class_name
        val_dest = val_dir / class_name
        test_dest = test_dir / class_name

        train_pairs = [(img, train_dest / img.name) for img in images[:train_count]]
        val_pairs = [(img, val_dest / img.name) for img in images[train_count : train_count + val_count]]
        test_pairs = [(img, test_dest / img.name) for img in images[train_count + val_count :]]

        for pairs in (train_pairs, val_pairs, test_pairs):
            dispatch_files(pairs, args.copy)

    print("Dataset split complete.")


if __name__ == "__main__":
    main()
