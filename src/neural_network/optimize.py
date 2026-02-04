"""Template optimizer runner for Etapa 6.

This script does not train models automatically. It generates a CSV template
for logging experiments so you can fill it after running train.py manually.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "optimization_experiments.csv"
    if csv_path.exists():
        print(f"Template already exists at {csv_path}")
        return

    rows = [
        {
            "exp": "Baseline",
            "change": "custom, 224x224, batch=16, lr=1e-3",
            "accuracy": "TBD",
            "f1_macro": "TBD",
            "train_time": "TBD",
            "notes": "Etapa 5 reference",
        },
        {
            "exp": "Exp1",
            "change": "mobilenet_v3_small",
            "accuracy": "TBD",
            "f1_macro": "TBD",
            "train_time": "TBD",
            "notes": "torchvision backbone",
        },
        {
            "exp": "Exp2",
            "change": "mobilenet_v3_small + pretrained",
            "accuracy": "TBD",
            "f1_macro": "TBD",
            "train_time": "TBD",
            "notes": "ImageNet init",
        },
        {
            "exp": "Exp3",
            "change": "batch=32, lr=5e-4",
            "accuracy": "TBD",
            "f1_macro": "TBD",
            "train_time": "TBD",
            "notes": "stability test",
        },
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote template: {csv_path} ({datetime.now().isoformat(timespec='seconds')})")


if __name__ == "__main__":
    main()
