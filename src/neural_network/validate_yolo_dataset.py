import argparse
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and labels")
    parser.add_argument("--data", type=str, default="config/fruitveg_detect.yaml")
    parser.add_argument("--root", type=str, default="data/yolo")
    return parser.parse_args()


def parse_names_from_yaml(path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not path.exists():
        return names
    lines = path.read_text(encoding="utf-8").splitlines()
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.strip():
                continue
            if not line.startswith(" "):
                break
            parts = line.strip().split(":", 1)
            if len(parts) != 2:
                continue
            idx = parts[0].strip()
            name = parts[1].strip().strip('"').strip("'")
            if idx.isdigit():
                names[int(idx)] = name
    return names


def validate_labels(label_path: Path, num_classes: int) -> List[str]:
    errors = []
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return errors
    for line_num, line in enumerate(content.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"{label_path}: line {line_num} should have 5 values")
            continue
        cls, x, y, w, h = parts
        if not cls.isdigit():
            errors.append(f"{label_path}: line {line_num} class id is not int")
            continue
        cls_id = int(cls)
        if num_classes and (cls_id < 0 or cls_id >= num_classes):
            errors.append(f"{label_path}: line {line_num} class id {cls_id} out of range")
        for val in (x, y, w, h):
            try:
                fval = float(val)
            except ValueError:
                errors.append(f"{label_path}: line {line_num} value {val} is not float")
                break
            if fval < 0.0 or fval > 1.0:
                errors.append(f"{label_path}: line {line_num} value {val} out of [0,1]")
    return errors


def main() -> None:
    args = parse_args()
    data_cfg = Path(args.data)
    root = Path(args.root)
    names = parse_names_from_yaml(data_cfg)
    num_classes = len(names)

    splits = ["train", "val", "test"]
    errors: List[str] = []
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            errors.append(f"Missing images dir: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"Missing labels dir: {lbl_dir}")
            continue
        images = [p for p in img_dir.rglob("*") if p.is_file()]
        for img in images:
            rel = img.relative_to(img_dir).with_suffix(".txt")
            label_path = lbl_dir / rel
            if not label_path.exists():
                errors.append(f"Missing label for image: {img}")
                continue
            errors.extend(validate_labels(label_path, num_classes))

    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("YOLO dataset validation passed.")


if __name__ == "__main__":
    main()
