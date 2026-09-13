#!/usr/bin/env python3
"""Train a YOLO detector on the Ultralytics Construction-PPE dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SmartVision PPE detector")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--name", default="smartvision-ppe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    train_kwargs = {
        "data": "construction-ppe.yaml",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(ROOT / "runs"),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device
    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    source = best if best.exists() else last
    if not source.exists():
        raise SystemExit(f"Training finished but no weights found in {save_dir}")
    target = MODELS / "best.pt"
    shutil.copy2(source, target)
    print(f"Copied {source} -> {target}")


if __name__ == "__main__":
    main()
