import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy classification images into YOLO raw folder")
    parser.add_argument("--sources", type=str, nargs="+", default=["data/train", "data/validation", "data/test"])
    parser.add_argument("--output", type=str, default="data/yolo_raw/images")
    parser.add_argument("--exts", type=str, nargs="+", default=[".jpg", ".jpeg", ".png"])
    return parser.parse_args()


def iter_images(folder: Path, exts: List[str]) -> Iterable[Path]:
    for ext in exts:
        yield from folder.rglob(f"*{ext}")


def build_target_name(img_path: Path, source_root: Path) -> str:
    rel = img_path.relative_to(source_root)
    class_name = rel.parts[0] if len(rel.parts) > 1 else "unknown"
    base = f"{class_name}_{img_path.stem}"
    digest = hashlib.md5(str(rel).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}{img_path.suffix.lower()}"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for src in args.sources:
        src_path = Path(src)
        if not src_path.exists():
            continue
        for img_path in iter_images(src_path, args.exts):
            target_name = build_target_name(img_path, src_path)
            target_path = output_dir / target_name
            shutil.copy2(img_path, target_path)
            total += 1

    print(f"Copied {total} images to {output_dir}")


if __name__ == "__main__":
    main()
