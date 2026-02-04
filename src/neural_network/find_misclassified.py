import argparse
from pathlib import Path
from typing import Dict, List

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find misclassified samples in YOLO dataset")
    parser.add_argument("--images-dir", type=str, default="data/yolo/images/val")
    parser.add_argument("--labels-dir", type=str, default="data/yolo/labels/val")
    parser.add_argument("--model", type=str, default="models/optimized_model.pt")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/misclassified_examples.csv")
    return parser.parse_args()


def load_label(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return None
    first = content.splitlines()[0].split()
    if not first:
        return None
    return int(first[0])


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    model_path = Path(args.model)

    if not images_dir.exists():
        raise SystemExit(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    classes_path = labels_dir / "classes.txt"
    classes = classes_path.read_text(encoding="utf-8").strip().splitlines()

    model = YOLO(str(model_path))

    misclassified: List[dict] = []
    images = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    for img_path in images:
        rel = img_path.relative_to(images_dir).with_suffix(".txt")
        label_path = labels_dir / rel
        true_id = load_label(label_path)
        if true_id is None:
            continue
        true_label = classes[true_id] if true_id < len(classes) else str(true_id)

        results = model.predict(source=str(img_path), conf=0.25, iou=0.45, max_det=1, verbose=False)
        if not results:
            continue
        result = results[0]
        if (
            result.boxes is None
            or result.boxes.cls is None
            or result.boxes.conf is None
            or len(result.boxes.cls) == 0
        ):
            continue
        pred_id = int(result.boxes.cls[0].item())
        pred_label = classes[pred_id] if pred_id < len(classes) else str(pred_id)
        conf = float(result.boxes.conf[0].item())

        if pred_id != true_id:
            reason = "încredere scăzută" if conf < 0.6 else "confuzie vizuală"
            misclassified.append(
                {
                    "image": str(img_path),
                    "true_label": true_label,
                    "predicted": pred_label,
                    "confidence": f"{conf:.3f}",
                    "note": reason,
                }
            )
            if len(misclassified) >= args.max_samples:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = "image,true_label,predicted,confidence,note\n"
    lines = [header]
    for item in misclassified:
        lines.append(
            f"{item['image']},{item['true_label']},{item['predicted']},{item['confidence']},{item['note']}\n"
        )
    output.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {len(misclassified)} misclassified samples to {output}")


if __name__ == "__main__":
    main()
