from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    MODELS_DIR,
)


def resolve_weights() -> Path:
    candidates = [
        MODELS_DIR / "best.pt",
        MODELS_DIR / "construction-ppe.pt",
        MODELS_DIR / "yolo11n-construction-ppe.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No trained SmartVision weights found in models/. "
        "Run: python scripts/train.py"
    )


class PPEDetector:
    def __init__(self, weights: Path | None = None, device: str | None = None) -> None:
        self.weights = Path(weights) if weights else resolve_weights()
        self.model = YOLO(str(self.weights))
        self.device = device
        names = getattr(self.model, "names", None) or CLASS_NAMES
        if isinstance(names, dict):
            self.names = {int(k): str(v) for k, v in names.items()}
        else:
            self.names = {i: str(n) for i, n in enumerate(names)}

    def predict(
        self,
        image: np.ndarray,
        conf: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(image, **kwargs)
        detections: list[dict[str, Any]] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            name = self.names.get(cls_id, str(cls_id))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": float(box.conf[0]),
                    "bbox": [float(v) for v in xyxy],
                }
            )
        return detections

    def annotate(self, image: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            name = det["class_name"]
            color = CLASS_COLORS.get(name, (255, 255, 255))
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            label = f"{name} {det['confidence']:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), bgr, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (12, 12, 12),
                1,
                cv2.LINE_AA,
            )
        return canvas
