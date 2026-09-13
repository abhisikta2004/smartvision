from app.alerts import build_alerts, summarize


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
