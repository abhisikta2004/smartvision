from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.alerts import build_alerts, summarize
from app.config import DEFAULT_CONFIDENCE, OUTPUTS_DIR, ROOT, STATIC_DIR, UPLOADS_DIR, VIDEO_FRAME_STRIDE
from app.detector import PPEDetector, resolve_weights
from app.zones import (
    ZoneMonitor,
    config_from_dict,
    draw_zone_overlays,
    empty_evaluation,
    load_zone_config,
    merge_zone_summary,
    save_zone_config,
)

app = FastAPI(title="SmartVision", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: PPEDetector | None = None
_load_error: str | None = None
_zone_config = load_zone_config()
_live_monitor = ZoneMonitor(_zone_config)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_detector() -> PPEDetector:
    global _detector, _load_error
    if _detector is not None:
        return _detector
    try:
        _detector = PPEDetector()
        _load_error = None
        return _detector
    except Exception as exc:  # noqa: BLE001
        _load_error = str(exc)
        raise HTTPException(status_code=503, detail=_load_error) from exc


def decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    return buf.tobytes()


@app.get("/api/health")
def health() -> dict[str, Any]:
    weights_ok = False
    weights_path = None
    try:
        path = resolve_weights()
        weights_ok = True
        weights_path = str(path)
    except FileNotFoundError:
        weights_ok = False
    return {
        "status": "ok" if weights_ok else "model_missing",
        "model_ready": weights_ok,
        "weights": weights_path,
        "error": _load_error,
        "danger_zones": len(_zone_config.zones),
    }


@app.get("/api/zones")
def get_zones() -> dict[str, Any]:
    return _zone_config.to_dict()


@app.put("/api/zones")
async def put_zones(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    global _zone_config
    try:
        config = config_from_dict(payload)
        save_zone_config(config)
        _zone_config = config
        _live_monitor.set_config(config)
        return config.to_dict()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid danger zone configuration: {exc}") from exc


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONFIDENCE),
    heuristics: str = Form("true"),
) -> JSONResponse:
    detector = get_detector()
    data = await file.read()
    image = decode_image(data)
    t0 = time.perf_counter()
    detections = detector.predict(image, conf=conf)
    alerts = build_alerts(detections, enable_heuristics=parse_bool(heuristics))
    zone_eval = ZoneMonitor(_zone_config).update(
        detections,
        int(image.shape[1]),
        int(image.shape[0]),
        now=time.time(),
        snapshot=True,
    )
    alerts = alerts + zone_eval.alerts
    annotated = detector.annotate(image, detections)
    annotated = draw_zone_overlays(annotated, _zone_config, zone_eval)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = OUTPUTS_DIR / out_name
    out_path.write_bytes(encode_jpeg(annotated))
    payload = {
        "source": file.filename,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "latency_ms": elapsed_ms,
        "detections": detections,
        "alerts": alerts,
        "summary": merge_zone_summary(summarize(detections, alerts), zone_eval),
        "danger_zone": zone_eval.to_status(),
        "annotated_url": f"/outputs/{out_name}",
    }
    return JSONResponse(payload)


@app.post("/api/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONFIDENCE),
    heuristics: str = Form("true"),
) -> JSONResponse:
    detector = get_detector()
    image = decode_image(await file.read())
    t0 = time.perf_counter()
    detections = detector.predict(image, conf=conf)
    alerts = build_alerts(detections, enable_heuristics=parse_bool(heuristics))
    zone_eval = _live_monitor.update(
        detections,
        int(image.shape[1]),
        int(image.shape[0]),
        now=time.time(),
        snapshot=False,
    )
    if zone_eval.new_alert_ids:
        evidence = draw_zone_overlays(detector.annotate(image, detections), _zone_config, zone_eval)
        out_name = f"zone-{uuid.uuid4().hex}.jpg"
        (OUTPUTS_DIR / out_name).write_bytes(encode_jpeg(evidence))
        evidence_url = f"/outputs/{out_name}"
        for alert in zone_eval.alerts:
            if alert["id"] in zone_eval.new_alert_ids:
                alert["evidence_url"] = evidence_url
    alerts = alerts + zone_eval.alerts
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse(
        {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "latency_ms": elapsed_ms,
            "detections": detections,
            "alerts": alerts,
            "summary": merge_zone_summary(summarize(detections, alerts), zone_eval),
            "danger_zone": zone_eval.to_status(),
        }
    )


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Form(DEFAULT_CONFIDENCE),
    heuristics: str = Form("true"),
    stride: int = Form(VIDEO_FRAME_STRIDE),
) -> JSONResponse:
    detector = get_detector()
    use_heuristics = parse_bool(heuristics)
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    src_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    src_path.write_bytes(await file.read())

    capture = cv2.VideoCapture(str(src_path))
    if not capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_name = f"{uuid.uuid4().hex}.mp4"
    out_path = OUTPUTS_DIR / out_name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    timeline: list[dict[str, Any]] = []
    total_counts: dict[str, int] = {}
    frame_index = 0
    processed = 0
    last_detections: list[dict[str, Any]] = []
    last_zone_eval = empty_evaluation()
    video_monitor = ZoneMonitor(_zone_config)
    t0 = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(1, stride) == 0:
            last_detections = detector.predict(frame, conf=conf)
            processed += 1
            alerts = build_alerts(last_detections, enable_heuristics=use_heuristics)
            last_zone_eval = video_monitor.update(
                last_detections,
                width,
                height,
                now=frame_index / fps,
                snapshot=False,
            )
            alerts = alerts + last_zone_eval.alerts
            for det in last_detections:
                name = det["class_name"]
                total_counts[name] = total_counts.get(name, 0) + 1
            if alerts:
                timestamp = frame_index / fps
                timeline.append(
                    {
                        "frame": frame_index,
                        "time": round(timestamp, 2),
                        "alerts": alerts,
                        "summary": merge_zone_summary(summarize(last_detections, alerts), last_zone_eval),
                    }
                )
        annotated = detector.annotate(frame, last_detections)
        annotated = draw_zone_overlays(annotated, _zone_config, last_zone_eval)
        writer.write(annotated)
        frame_index += 1

    capture.release()
    writer.release()

    unique_alerts: dict[str, dict[str, Any]] = {}
    for event in timeline:
        for alert in event["alerts"]:
            key = alert["type"]
            extra = [alert.get("worker_id"), alert.get("zone_name")]
            if any(extra):
                key = f"{key}:{alert.get('worker_id')}:{alert.get('zone_name')}"
            if key not in unique_alerts:
                unique_alerts[key] = {**alert, "first_seen": event["time"], "occurrences": 0}
            unique_alerts[key]["occurrences"] += 1

    alerts_list = list(unique_alerts.values())
    dummy_dets = [{"class_name": name} for name, count in total_counts.items() for _ in range(min(count, 3))]
    summary = merge_zone_summary(summarize(dummy_dets, alerts_list), last_zone_eval)
    summary["frames"] = frame_index
    summary["processed_frames"] = processed
    summary["class_counts"] = total_counts

    return JSONResponse(
        {
            "source": file.filename,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "fps": fps,
            "width": width,
            "height": height,
            "alerts": alerts_list,
            "timeline": timeline[:200],
            "summary": summary,
            "danger_zone": last_zone_eval.to_status(),
            "annotated_url": f"/outputs/{out_name}",
        }
    )


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
