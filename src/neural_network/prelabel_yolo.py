import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-generate YOLO boxes using a pretrained model")
    parser.add_argument("--images-dir", type=str, default="data/yolo_raw/images")
    parser.add_argument("--labels-dir", type=str, default="data/yolo_raw/labels")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")

    model = YOLO(args.model)
    split_dirs = ["train", "val", "test"]
    has_split_dirs = all((images_dir / split).exists() for split in split_dirs)

    if has_split_dirs:
        target_splits = split_dirs
    else:
        target_splits = [""]

    total_images = 0
    for split in target_splits:
        split_images_dir = images_dir / split if split else images_dir
        split_labels_dir = labels_dir / split if split else labels_dir
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        images = (
            list(split_images_dir.rglob("*.jpg"))
            + list(split_images_dir.rglob("*.jpeg"))
            + list(split_images_dir.rglob("*.png"))
        )
        if not images:
            continue
        total_images += len(images)

        for img_path in images:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            if result.boxes is None or result.boxes.xyxy is None:
                continue

            w = result.orig_shape[1]
            h = result.orig_shape[0]
            label_lines = []
            for (x1, y1, x2, y2) in result.boxes.xyxy.cpu().tolist():
                x_center = ((x1 + x2) / 2.0) / w
                y_center = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            if label_lines:
                rel_path = img_path.relative_to(split_images_dir).with_suffix(".txt")
                out_path = split_labels_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("\n".join(label_lines), encoding="utf-8")

    if total_images == 0:
        raise SystemExit("No images found for prelabeling.")

    classes_path = labels_dir.parent / "classes.txt"
    classes_path.write_text("object\n", encoding="utf-8")
    print("Prelabeling done. Review boxes in LabelImg and rename labels to real classes.")


if __name__ == "__main__":
    main()
