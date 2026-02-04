import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import extra real-world images into an ImageFolder class directory.",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Folder containing images to import (can contain subfolders).",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination class folder, e.g. data/train/banana",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max images to import (0 = no limit).",
    )
    parser.add_argument(
        "--prefix",
        default="real_",
        help="Filename prefix added to imported images.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dest = Path(args.dest)

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {src}")

    dest.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in src.rglob("*") if p.is_file() and is_image(p)]
    candidates.sort()

    limit = args.max if args.max and args.max > 0 else None
    imported = 0

    for idx, path in enumerate(candidates):
        if limit is not None and imported >= limit:
            break

        target_name = f"{args.prefix}{idx:05d}{path.suffix.lower()}"
        target = dest / target_name

        if args.move:
            shutil.move(str(path), str(target))
        else:
            shutil.copy2(str(path), str(target))

        imported += 1

    print(f"Imported {imported} images into: {dest}")


if __name__ == "__main__":
    main()
