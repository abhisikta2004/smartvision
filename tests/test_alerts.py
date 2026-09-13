from pathlib import Path

import numpy as np

from app.alerts import build_alerts, summarize
from app.zones import (
    DangerZone,
    ZoneConfig,
    ZoneMonitor,
    config_from_dict,
    load_zone_config,
    missing_required_ppe,
    point_in_polygon,
    zone_points_px,
)


def _person(bbox, name="Person"):
    return {"class_name": name, "confidence": 0.9, "bbox": bbox}


def _ppe(name, bbox):
    return {"class_name": name, "confidence": 0.9, "bbox": bbox}


def _config(zones, threshold=2.0):
    return ZoneConfig(violation_threshold_seconds=threshold, alert_cooldown_seconds=30, track_lost_seconds=1.5, zones=zones)


SQUARE = DangerZone(
    id="zone_1",
    name="Construction Area",
    points=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
    required_ppe=["helmet", "vest"],
)

WELDING = DangerZone(
    id="zone_2",
    name="Welding Area",
    points=[[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]],
    required_ppe=["helmet", "gloves", "goggles", "vest"],
)


def test_missing_helmet_raises_critical_alert():
    detections = [
        {"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        {"class_name": "no_helmet", "confidence": 0.8, "bbox": [1, 1, 4, 4]},
    ]
    alerts = build_alerts(detections, enable_heuristics=False)
    assert alerts[0]["type"] == "no_helmet"
    assert summarize(detections, alerts)["status"] == "violation"


def test_person_without_vest_uses_heuristic():
    detections = [{"class_name": "Person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    alerts = build_alerts(detections, enable_heuristics=True)
    types = {a["type"] for a in alerts}
    assert "missing_vest" in types
    assert "possible_no_helmet" in types


def test_foot_point_inside_polygon():
    contour = zone_points_px(SQUARE, 100, 100)
    assert point_in_polygon((50, 80), contour)
    assert not point_in_polygon((5, 5), contour)


def test_outside_zone_complete_ppe_no_zone_alert():
    monitor = ZoneMonitor(_config([SQUARE]))
    detections = [
        _person([2, 2, 16, 18]),
        _ppe("helmet", [12, 10, 20, 18]),
        _ppe("vest", [12, 18, 28, 36]),
    ]
    result = monitor.update(detections, 100, 100, now=0, snapshot=False)
    assert result.workers_in_zones == 0
    assert result.alerts == []


def test_outside_zone_missing_ppe_keeps_existing_warning():
    detections = [_person([2, 2, 16, 18]), _ppe("no_helmet", [4, 2, 12, 8])]
    alerts = build_alerts(detections, enable_heuristics=False)
    monitor = ZoneMonitor(_config([SQUARE]))
    zone = monitor.update(detections, 100, 100, now=3, snapshot=False)
    assert alerts[0]["type"] == "no_helmet"
    assert zone.alerts == []


def test_inside_zone_complete_required_ppe_no_critical():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    detections = [
        _person([40, 40, 70, 80]),
        _ppe("helmet", [45, 40, 55, 50]),
        _ppe("vest", [42, 50, 68, 75]),
    ]
    result = monitor.update(detections, 100, 100, now=5, snapshot=False)
    assert result.workers_in_zones == 1
    assert result.alerts == []
    assert result.active_critical_violations == 0


def test_inside_zone_missing_ppe_under_threshold_no_alert():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    detections = [_person([40, 40, 70, 80]), _ppe("no_helmet", [45, 40, 55, 50])]
    first = monitor.update(detections, 100, 100, now=1.0, snapshot=False)
    second = monitor.update(detections, 100, 100, now=2.5, snapshot=False)
    assert first.alerts == []
    assert second.alerts == []
    assert second.occupants[0].pending is True


def test_inside_zone_missing_ppe_over_threshold_critical_alert():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    detections = [_person([40, 40, 70, 80]), _ppe("vest", [42, 50, 68, 75])]
    monitor.update(detections, 100, 100, now=1.0)
    monitor.update(detections, 100, 100, now=2.0)
    result = monitor.update(detections, 100, 100, now=3.1)
    assert result.alerts
    assert result.alerts[0]["type"] == "danger_zone_violation"
    assert result.alerts[0]["severity"] == "critical"
    assert result.alerts[0]["zone_name"] == "Construction Area"
    assert "helmet" in result.alerts[0]["missing_items"]


def test_leaving_zone_resets_violation():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    inside = [{"class_name": "Person", "confidence": 0.9, "bbox": [40, 40, 70, 80], "track_id": 12}]
    outside = [{"class_name": "Person", "confidence": 0.9, "bbox": [2, 2, 16, 18], "track_id": 12}]
    monitor.update(inside, 100, 100, now=1.0)
    monitor.update(inside, 100, 100, now=2.5)
    left = monitor.update(outside, 100, 100, now=3.0)
    assert left.workers_in_zones == 0
    back = monitor.update(inside, 100, 100, now=3.5)
    assert back.alerts == []
    monitor.update(inside, 100, 100, now=4.5)
    later = monitor.update(inside, 100, 100, now=5.6)
    assert later.alerts


def test_multiple_workers_evaluated_independently():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    detections = [
        _person([40, 40, 55, 80]),
        _person([60, 40, 78, 80]),
        _ppe("helmet", [42, 40, 50, 50]),
        _ppe("vest", [42, 50, 54, 75]),
    ]
    monitor.update(detections, 100, 100, now=0)
    monitor.update(detections, 100, 100, now=1.0)
    result = monitor.update(detections, 100, 100, now=2.1)
    assert result.workers_in_zones == 2
    assert result.active_critical_violations == 1
    assert len(result.alerts) == 1


def test_multiple_zones_use_matching_required_ppe():
    monitor = ZoneMonitor(_config([SQUARE, WELDING], threshold=0))
    detections = [
        _person([10, 10, 30, 30]),
        _ppe("helmet", [12, 10, 20, 16]),
        _ppe("vest", [12, 16, 28, 28]),
    ]
    result = monitor.update(detections, 100, 100, now=1, snapshot=True)
    assert result.alerts
    missing = result.alerts[0]["missing_items"]
    assert "gloves" in missing
    assert "goggles" in missing
    assert result.alerts[0]["zone_name"] == "Welding Area"


def test_same_worker_does_not_alert_every_frame():
    monitor = ZoneMonitor(_config([SQUARE], threshold=2))
    detections = [_person([40, 40, 70, 80])]
    monitor.update(detections, 100, 100, now=0)
    monitor.update(detections, 100, 100, now=1.0)
    first = monitor.update(detections, 100, 100, now=2.1)
    second = monitor.update(detections, 100, 100, now=2.2)
    third = monitor.update(detections, 100, 100, now=2.3)
    assert len(first.alerts) == 1
    assert second.alerts == []
    assert third.alerts == []


def test_invalid_and_missing_zone_config_does_not_crash():
    empty = load_zone_config(Path("/tmp/does-not-exist-danger-zones.json"))
    assert empty.zones == []
    parsed = config_from_dict({"zones": [{"name": "bad", "points": [[1, 1]]}]})
    assert parsed.zones == []
    monitor = ZoneMonitor(parsed)
    result = monitor.update([_person([1, 1, 10, 10])], 100, 100, now=3)
    assert result.alerts == []


def test_missing_required_ppe_uses_person_overlap():
    person = [0, 0, 50, 50]
    other = [80, 80, 90, 90]
    detections = [_person(person), _ppe("helmet", other), _ppe("vest", [5, 20, 20, 40])]
    missing = missing_required_ppe(person, detections, ["helmet", "vest"])
    assert missing == ["helmet"]


def test_draw_handles_empty_frame():
    from app.zones import draw_zone_overlays, empty_evaluation

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    out = draw_zone_overlays(frame, ZoneConfig(), empty_evaluation())
    assert out.shape == frame.shape
