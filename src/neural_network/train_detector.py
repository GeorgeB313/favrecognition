import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    # Parametri standard pentru antrenarea detectorului YOLO.
    parser = argparse.ArgumentParser(description="Train YOLO detector for fruits/vegetables")
    parser.add_argument("--data", type=str, default="config/fruitveg_detect.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="fruitveg")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--csv-path", type=str, default="results/detect_training.csv")
    return parser.parse_args()


def main() -> None:
    # Antrenează modelul și copiază rezultatele în results/ și models/.
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        verbose=True,
    )

    project_dir = Path(args.project)
    best_candidates = list(project_dir.rglob("weights/best.pt"))
    latest_best = max(best_candidates, key=lambda p: p.stat().st_mtime, default=None)

    results_candidates = list(project_dir.rglob("results.csv"))
    latest_results = max(results_candidates, key=lambda p: p.stat().st_mtime, default=None)

    if latest_results:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_results, csv_path)
        print(f"Saved training CSV to {csv_path}")
    else:
        print(f"Training CSV not found under {project_dir}")

    if latest_best:
        output = Path("models") / "fruitveg_detector.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_best, output)
        print(f"Saved detector to {output}")
    else:
        print(f"Training finished, but best.pt not found under {project_dir}.")


if __name__ == "__main__":
    main()
