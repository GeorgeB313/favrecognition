import argparse
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full-image YOLO boxes per image")
    parser.add_argument("--images-dir", type=str, default="data/yolo/images")
    parser.add_argument("--labels-dir", type=str, default="data/yolo/labels")
    parser.add_argument("--data", type=str, default="config/fruitveg_detect.yaml")
    parser.add_argument("--overwrite", action="store_true")
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


def write_classes_txt(base_dir: Path, class_names: List[str]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    classes_path = base_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    data_cfg = Path(args.data)

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")

    names = parse_names_from_yaml(data_cfg)
    if not names:
        raise SystemExit(f"No class names found in {data_cfg}")

    id_to_name = [name for _, name in sorted(names.items(), key=lambda item: item[0])]
    name_to_id = {name: idx for idx, name in enumerate(id_to_name)}

    splits = ["train", "val", "test"]
    total = 0
    for split in splits:
        split_images_dir = images_dir / split
        if not split_images_dir.exists():
            continue
        split_labels_dir = labels_dir / split
        write_classes_txt(split_labels_dir, id_to_name)

        for class_dir in split_images_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in name_to_id:
                continue
            class_id = name_to_id[class_name]

            class_label_dir = split_labels_dir / class_name
            write_classes_txt(class_label_dir, id_to_name)

            for img_path in class_dir.rglob("*"):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                rel = img_path.relative_to(split_images_dir).with_suffix(".txt")
                out_path = split_labels_dir / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not args.overwrite:
                    continue
                out_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
                total += 1

    print(f"Generated {total} full-image labels.")


if __name__ == "__main__":
    main()
