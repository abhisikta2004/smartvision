from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from app.alerts import Alert
from app.config import (
    DANGER_ZONE_ALERT_COOLDOWN_SECONDS,
    DANGER_ZONE_TRACK_LOST_SECONDS,
    DANGER_ZONE_VIOLATION_SECONDS,
    DANGER_ZONES_PATH,
    DEFAULT_ZONE_REQUIRED_PPE,
    MISSING_PPE_BY_REQUIRED,
    PERSON_CLASS,
    PPE_NAME_ALIASES,
    WORN_PPE,
)

logger = logging.getLogger(__name__)


@dataclass
class DangerZone:
    id: str
    name: str
    points: list[list[float]]
    required_ppe: list[str] | None = None
    severity: str = "critical"
    violation_threshold_seconds: float | None = None
    frame_width: int | None = None
    frame_height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZoneConfig:
    violation_threshold_seconds: float = DANGER_ZONE_VIOLATION_SECONDS
    alert_cooldown_seconds: float = DANGER_ZONE_ALERT_COOLDOWN_SECONDS
    track_lost_seconds: float = DANGER_ZONE_TRACK_LOST_SECONDS
    default_required_ppe: list[str] = field(default_factory=lambda: list(DEFAULT_ZONE_REQUIRED_PPE))
    zones: list[DangerZone] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_threshold_seconds": self.violation_threshold_seconds,
            "alert_cooldown_seconds": self.alert_cooldown_seconds,
            "track_lost_seconds": self.track_lost_seconds,
            "default_required_ppe": self.default_required_ppe,
            "zones": [z.to_dict() for z in self.zones],
        }


@dataclass
class Occupant:
    track_id: int
    bbox: list[float]
    foot: list[float]
    zone_id: str | None
    zone_name: str | None
    missing_ppe: list[str]
    pending: bool
    confirmed: bool


@dataclass
class ZoneEvaluation:
    occupants: list[Occupant] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    workers_in_zones: int = 0
    active_critical_violations: int = 0
    by_zone: dict[str, int] = field(default_factory=dict)
    new_alert_ids: list[str] = field(default_factory=list)

    def to_status(self) -> dict[str, Any]:
        return {
            "workers_in_zones": self.workers_in_zones,
            "active_critical_violations": self.active_critical_violations,
            "by_zone": self.by_zone,
            "occupants": [asdict(item) for item in self.occupants],
        }


@dataclass
class _Track:
    track_id: int
    bbox: list[float]
    last_seen: float
    violation_accum: dict[str, float] = field(default_factory=dict)
    last_alert_at: dict[str, float] = field(default_factory=dict)


def normalize_ppe_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    return PPE_NAME_ALIASES.get(key, key)


def normalize_required_ppe(items: list[str] | None, default: list[str]) -> list[str]:
    if items is None:
        source = default
    else:
        source = items
    seen: list[str] = []
    for item in source:
        name = normalize_ppe_name(item)
        if name and name not in seen:
            seen.append(name)
    return seen


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def foot_point(bbox: list[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def ppe_belongs_to_person(ppe_bbox: list[float], person_bbox: list[float]) -> bool:
    x1, y1, x2, y2 = person_bbox
    cx = (ppe_bbox[0] + ppe_bbox[2]) / 2.0
    cy = (ppe_bbox[1] + ppe_bbox[3]) / 2.0
    if x1 <= cx <= x2 and y1 <= cy <= y2:
        return True
    return iou(ppe_bbox, person_bbox) > 0.05


def zone_points_px(zone: DangerZone, width: int, height: int) -> np.ndarray | None:
    if not zone.points or len(zone.points) < 3:
        return None
    try:
        pts = [(float(p[0]), float(p[1])) for p in zone.points]
    except (TypeError, ValueError, IndexError):
        return None
    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)
    if max_x <= 1.5 and max_y <= 1.5:
        scaled = [(p[0] * width, p[1] * height) for p in pts]
    else:
        ref_w = zone.frame_width or width
        ref_h = zone.frame_height or height
        if ref_w <= 0 or ref_h <= 0:
            scaled = pts
        else:
            scaled = [(p[0] * width / ref_w, p[1] * height / ref_h) for p in pts]
    contour = np.array(scaled, dtype=np.float32)
    if cv2.contourArea(contour) <= 1:
        return None
    return contour


def point_in_polygon(point: tuple[float, float], contour: np.ndarray) -> bool:
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0


def missing_required_ppe(
    person_bbox: list[float],
    detections: list[dict[str, Any]],
    required: list[str],
) -> list[str]:
    worn: set[str] = set()
    missing_labels: set[str] = set()
    for det in detections:
        name = det.get("class_name")
        bbox = det.get("bbox")
        if not name or not bbox or name == PERSON_CLASS:
            continue
        if not ppe_belongs_to_person(bbox, person_bbox):
            continue
        if name in WORN_PPE:
            worn.add(name)
        if name in MISSING_PPE_BY_REQUIRED.values():
            missing_labels.add(name)

    missing: list[str] = []
    for item in required:
        no_label = MISSING_PPE_BY_REQUIRED.get(item)
        if no_label and no_label in missing_labels:
            missing.append(item)
            continue
        worn_name = item if item in WORN_PPE else None
        if worn_name and worn_name not in worn:
            missing.append(item)
    return missing


def _parse_zone(raw: dict[str, Any], index: int) -> DangerZone | None:
    points = raw.get("points") or []
    if not isinstance(points, list) or len(points) < 3:
        logger.warning("Skipping danger zone %s: need at least 3 polygon points", raw.get("id") or index)
        return None
    cleaned: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            logger.warning("Skipping danger zone with invalid vertex: %s", raw.get("id") or index)
            return None
        cleaned.append([float(point[0]), float(point[1])])
    required = raw.get("required_ppe")
    if required is not None and not isinstance(required, list):
        required = None
    return DangerZone(
        id=str(raw.get("id") or f"zone_{index + 1}"),
        name=str(raw.get("name") or f"Danger Zone {index + 1}"),
        points=cleaned,
        required_ppe=[str(item) for item in required] if required is not None else None,
        severity=str(raw.get("severity") or "critical"),
        violation_threshold_seconds=(
            float(raw["violation_threshold_seconds"]) if raw.get("violation_threshold_seconds") is not None else None
        ),
        frame_width=int(raw["frame_width"]) if raw.get("frame_width") else None,
        frame_height=int(raw["frame_height"]) if raw.get("frame_height") else None,
    )


def config_from_dict(data: dict[str, Any] | None) -> ZoneConfig:
    data = data or {}
    zones: list[DangerZone] = []
    raw_zones = data.get("zones") or []
    if isinstance(raw_zones, list):
        for index, raw in enumerate(raw_zones):
            if not isinstance(raw, dict):
                continue
            zone = _parse_zone(raw, index)
            if zone:
                zones.append(zone)
    default_ppe = data.get("default_required_ppe", list(DEFAULT_ZONE_REQUIRED_PPE))
    if not isinstance(default_ppe, list):
        default_ppe = list(DEFAULT_ZONE_REQUIRED_PPE)
    return ZoneConfig(
        violation_threshold_seconds=float(data.get("violation_threshold_seconds", DANGER_ZONE_VIOLATION_SECONDS)),
        alert_cooldown_seconds=float(data.get("alert_cooldown_seconds", DANGER_ZONE_ALERT_COOLDOWN_SECONDS)),
        track_lost_seconds=float(data.get("track_lost_seconds", DANGER_ZONE_TRACK_LOST_SECONDS)),
        default_required_ppe=[str(item) for item in default_ppe],
        zones=zones,
    )


def load_zone_config(path=DANGER_ZONES_PATH) -> ZoneConfig:
    try:
        if not path.exists():
            return ZoneConfig()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            logger.warning("Danger zone config is not an object; ignoring")
            return ZoneConfig()
        return config_from_dict(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load danger zone config: %s", exc)
        return ZoneConfig()


def save_zone_config(config: ZoneConfig, path=DANGER_ZONES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")


def empty_evaluation() -> ZoneEvaluation:
    return ZoneEvaluation()


def _zones_containing(foot: tuple[float, float], zones: list[DangerZone], width: int, height: int) -> list[tuple[DangerZone, np.ndarray]]:
    found: list[tuple[DangerZone, np.ndarray]] = []
    for zone in zones:
        contour = zone_points_px(zone, width, height)
        if contour is None:
            continue
        if point_in_polygon(foot, contour):
            found.append((zone, contour))
    return found


def _alert_key(track_id: int, zone_id: str, missing: list[str]) -> str:
    return f"{track_id}:{zone_id}:{','.join(missing)}"


def _make_alert(track_id: int, zone: DangerZone, missing: list[str], now: float) -> dict[str, Any]:
    worker = f"#{track_id}"
    missing_txt = ", ".join(item.replace("_", " ").title() for item in missing) if missing else "unauthorized entry"
    stamp = time_label(now)
    alert = Alert(
        id=str(uuid4()),
        type="danger_zone_violation",
        severity=zone.severity or "critical",
        title="CRITICAL SAFETY ALERT",
        message=(
            f"Worker {worker} in {zone.name}. Missing PPE: {missing_txt}."
            if missing
            else f"Worker {worker} entered restricted zone {zone.name}."
        ),
        count=1,
        worker_id=str(track_id),
        zone_name=zone.name,
        missing_items=missing,
        timestamp=stamp,
    )
    return alert.to_dict()


def time_label(now: float) -> str:
    if now > 1_000_000_000:
        from datetime import datetime

        return datetime.fromtimestamp(now).strftime("%H:%M:%S")
    minutes, seconds = divmod(int(now), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ZoneMonitor:
    """Tracks workers across frames and emits persistent danger-zone alerts."""

    def __init__(self, config: ZoneConfig | None = None) -> None:
        self.config = config or ZoneConfig()
        self.tracks: dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def set_config(self, config: ZoneConfig) -> None:
        self.config = config

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        now: float,
        snapshot: bool = False,
    ) -> ZoneEvaluation:
        people = [d for d in detections if d.get("class_name") == PERSON_CLASS and d.get("bbox")]
        assigned = self._assign_tracks(people, now)
        evaluation = ZoneEvaluation()

        for person, track in assigned:
            dt = max(0.0, now - track.last_seen)
            if dt > self.config.track_lost_seconds:
                track.violation_accum.clear()
                dt = 0.0
            track.bbox = list(person["bbox"])
            person["track_id"] = track.track_id
            foot = foot_point(track.bbox)
            inside = _zones_containing(foot, self.config.zones, frame_width, frame_height)
            active_keys: set[str] = set()
            if not inside:
                track.violation_accum.clear()
                track.last_seen = now
                evaluation.occupants.append(
                    Occupant(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        foot=[foot[0], foot[1]],
                        zone_id=None,
                        zone_name=None,
                        missing_ppe=[],
                        pending=False,
                        confirmed=False,
                    )
                )
                continue

            evaluation.workers_in_zones += 1
            for zone, _contour in inside:
                required = normalize_required_ppe(zone.required_ppe, self.config.default_required_ppe)
                missing = missing_required_ppe(track.bbox, detections, required)
                in_violation = bool(missing) or required == []
                key = _alert_key(track.track_id, zone.id, missing)
                pending = False
                confirmed = False
                if in_violation:
                    active_keys.add(key)
                    elapsed = track.violation_accum.get(key, 0.0) + dt
                    if snapshot:
                        elapsed = max(elapsed, self.config.violation_threshold_seconds)
                    track.violation_accum[key] = elapsed
                    threshold = zone.violation_threshold_seconds
                    if threshold is None:
                        threshold = self.config.violation_threshold_seconds
                    pending = elapsed < threshold
                    confirmed = elapsed >= threshold
                    if confirmed:
                        evaluation.active_critical_violations += 1
                        evaluation.by_zone[zone.name] = evaluation.by_zone.get(zone.name, 0) + 1
                        last_alert = track.last_alert_at.get(key)
                        cooled = last_alert is None or (now - last_alert) >= self.config.alert_cooldown_seconds
                        if cooled:
                            alert = _make_alert(track.track_id, zone, missing, now)
                            evaluation.alerts.append(alert)
                            evaluation.new_alert_ids.append(alert["id"])
                            track.last_alert_at[key] = now
                evaluation.occupants.append(
                    Occupant(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        foot=[foot[0], foot[1]],
                        zone_id=zone.id,
                        zone_name=zone.name,
                        missing_ppe=missing,
                        pending=pending,
                        confirmed=confirmed,
                    )
                )

            stale = [key for key in track.violation_accum if key not in active_keys]
            for key in stale:
                track.violation_accum.pop(key, None)
            track.last_seen = now

        lost_after = self.config.track_lost_seconds
        stale_tracks = [tid for tid, track in self.tracks.items() if now - track.last_seen > lost_after]
        for tid in stale_tracks:
            self.tracks.pop(tid, None)
        return evaluation

    def _assign_tracks(self, people: list[dict[str, Any]], now: float) -> list[tuple[dict[str, Any], _Track]]:
        remaining = [t for t in self.tracks.values() if now - t.last_seen <= self.config.track_lost_seconds]
        used: set[int] = set()
        assigned: list[tuple[dict[str, Any], _Track]] = []
        for person in people:
            explicit = person.get("track_id")
            if explicit is not None:
                track = self.tracks.get(int(explicit))
                if track is None:
                    track = _Track(track_id=int(explicit), bbox=list(person["bbox"]), last_seen=now)
                    self.tracks[track.track_id] = track
                    self._next_id = max(self._next_id, track.track_id + 1)
                used.add(track.track_id)
                assigned.append((person, track))
                continue
            best: _Track | None = None
            best_iou = 0.3
            for track in remaining:
                if track.track_id in used:
                    continue
                score = iou(person["bbox"], track.bbox)
                if score > best_iou:
                    best_iou = score
                    best = track
            if best is None:
                best = _Track(track_id=self._next_id, bbox=list(person["bbox"]), last_seen=now)
                self.tracks[best.track_id] = best
                self._next_id += 1
            used.add(best.track_id)
            assigned.append((person, best))
        return assigned


def draw_zone_overlays(
    image: np.ndarray,
    config: ZoneConfig,
    evaluation: ZoneEvaluation | None = None,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    contours: list[np.ndarray] = []
    labels: list[tuple[DangerZone, np.ndarray]] = []
    for zone in config.zones:
        contour = zone_points_px(zone, width, height)
        if contour is None:
            continue
        contours.append(contour.astype(np.int32))
        labels.append((zone, contour))
    if contours:
        cv2.fillPoly(overlay, contours, (42, 90, 255))
        canvas = cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0)
        for zone, contour in labels:
            pts = contour.astype(np.int32)
            cv2.polylines(canvas, [pts], True, (42, 90, 255), 2, cv2.LINE_AA)
            cx, cy = int(pts[:, 0, 0].mean()), int(pts[:, 0, 1].mean())
            label = f"DANGER ZONE · {zone.name}"
            cv2.putText(canvas, label, (max(8, cx - 80), max(18, cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 240, 255), 2, cv2.LINE_AA)

    occupants = evaluation.occupants if evaluation else []
    for occupant in occupants:
        if not occupant.zone_name:
            continue
        x1, y1, x2, y2 = [int(v) for v in occupant.bbox]
        color = (42, 90, 255) if occupant.confirmed or occupant.missing_ppe else (52, 152, 219)
        if occupant.confirmed:
            color = (36, 36, 220)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        fx, fy = int(occupant.foot[0]), int(occupant.foot[1])
        cv2.circle(canvas, (fx, fy), 5, color, -1)
        missing = ", ".join(item.replace("_", " ").title() for item in occupant.missing_ppe) or "—"
        lines = [
            "CRITICAL" if occupant.confirmed else ("IN ZONE" if not occupant.missing_ppe else "ZONE CHECK"),
            f"Worker #{occupant.track_id}",
            f"Missing: {missing}",
            f"Danger Zone: {occupant.zone_name}",
        ]
        ty = max(16, y1 - 8)
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 8, ty + 2), color, -1)
            cv2.putText(canvas, line, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
            ty -= th + 8
    return canvas


def merge_zone_summary(summary: dict[str, Any], evaluation: ZoneEvaluation) -> dict[str, Any]:
    summary = dict(summary)
    summary["workers_in_zones"] = evaluation.workers_in_zones
    summary["active_critical_violations"] = evaluation.active_critical_violations
    summary["zone_violations"] = evaluation.by_zone
    if evaluation.active_critical_violations:
        summary["status"] = "violation"
    return summary
