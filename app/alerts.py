from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from uuid import uuid4

from app.config import MISSING_PPE, PERSON_CLASS, VIOLATION_META, WORN_PPE


@dataclass
class Alert:
    id: str
    type: str
    severity: str
    title: str
    message: str
    count: int
    worker_id: str | None = None
    zone_name: str | None = None
    missing_items: list[str] | None = None
    timestamp: str | None = None
    evidence_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def _add_alert(alerts: dict[str, Alert], key: str, count: int = 1) -> None:
    meta = VIOLATION_META[key]
    if key in alerts:
        alerts[key].count += count
        return
    alerts[key] = Alert(
        id=str(uuid4()),
        type=key,
        severity=meta["severity"],
        title=meta["title"],
        message=meta["message"],
        count=count,
    )


def build_alerts(detections: list[dict[str, Any]], enable_heuristics: bool = True) -> list[dict[str, Any]]:
    alerts: dict[str, Alert] = {}
    counts: dict[str, int] = {}
    for det in detections:
        name = det["class_name"]
        counts[name] = counts.get(name, 0) + 1
        if name in MISSING_PPE:
            _add_alert(alerts, name)

    person_count = counts.get(PERSON_CLASS, 0)
    if enable_heuristics and person_count > 0:
        if counts.get("vest", 0) == 0:
            _add_alert(alerts, "missing_vest")
        helmet_signals = counts.get("helmet", 0) + counts.get("no_helmet", 0)
        if helmet_signals == 0:
            _add_alert(alerts, "possible_no_helmet")

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ordered = sorted(alerts.values(), key=lambda a: (severity_rank.get(a.severity, 9), a.type))
    return [alert.to_dict() for alert in ordered]


def summarize(detections: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    people = sum(1 for d in detections if d["class_name"] == PERSON_CLASS)
    worn = sum(1 for d in detections if d["class_name"] in WORN_PPE)
    missing = sum(1 for d in detections if d["class_name"] in MISSING_PPE)
    critical = sum(1 for a in alerts if a["severity"] == "critical")
    status = "compliant"
    if critical or missing:
        status = "violation"
    elif alerts:
        status = "warning"
    elif people == 0:
        status = "idle"
    return {
        "people": people,
        "worn_ppe": worn,
        "missing_ppe": missing,
        "alert_count": len(alerts),
        "status": status,
    }
