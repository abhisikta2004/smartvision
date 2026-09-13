from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
UPLOADS_DIR = ROOT / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"
CONFIG_DIR = ROOT / "config"
DANGER_ZONES_PATH = CONFIG_DIR / "danger_zones.json"

for directory in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

WORN_PPE = {"helmet", "gloves", "vest", "boots", "goggles"}
MISSING_PPE = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}
PERSON_CLASS = "Person"

VIOLATION_META = {
    "no_helmet": {
        "severity": "critical",
        "title": "Helmet missing",
        "message": "Worker detected without a hard hat.",
    },
    "no_goggle": {
        "severity": "high",
        "title": "Eye protection missing",
        "message": "Worker detected without safety goggles.",
    },
    "no_gloves": {
        "severity": "high",
        "title": "Gloves missing",
        "message": "Worker detected without protective gloves.",
    },
    "no_boots": {
        "severity": "high",
        "title": "Safety boots missing",
        "message": "Worker detected without safety boots.",
    },
    "missing_vest": {
        "severity": "critical",
        "title": "Hi-vis vest not detected",
        "message": "A person is in frame but no safety vest was detected.",
    },
    "possible_no_helmet": {
        "severity": "medium",
        "title": "Helmet not confirmed",
        "message": "A person is in frame but no helmet was detected.",
    },
    "danger_zone_violation": {
        "severity": "critical",
        "title": "CRITICAL SAFETY ALERT",
        "message": "Worker inside a danger zone without required PPE.",
    },
}

CLASS_COLORS = {
    "helmet": (46, 204, 113),
    "gloves": (39, 174, 96),
    "vest": (241, 196, 15),
    "boots": (52, 152, 219),
    "goggles": (26, 188, 156),
    "Person": (52, 152, 219),
    "none": (149, 165, 166),
    "no_helmet": (231, 76, 60),
    "no_goggle": (192, 57, 43),
    "no_gloves": (230, 126, 34),
    "no_boots": (211, 84, 0),
}

DEFAULT_CONFIDENCE = 0.35
DEFAULT_IOU = 0.45
VIDEO_FRAME_STRIDE = 3
HEURISTIC_PERSON_AREA_RATIO = 0.02
DANGER_ZONE_VIOLATION_SECONDS = 2.0
DANGER_ZONE_ALERT_COOLDOWN_SECONDS = 20.0
DANGER_ZONE_TRACK_LOST_SECONDS = 1.5
DEFAULT_ZONE_REQUIRED_PPE = ["helmet", "vest", "boots"]
PPE_NAME_ALIASES = {
    "helmet": "helmet",
    "hard_hat": "helmet",
    "hardhat": "helmet",
    "vest": "vest",
    "hi-vis": "vest",
    "hivis": "vest",
    "gloves": "gloves",
    "goggles": "goggles",
    "goggle": "goggles",
    "boots": "boots",
    "safety_shoes": "boots",
    "safety-shoes": "boots",
    "shoes": "boots",
}
MISSING_PPE_BY_REQUIRED = {
    "helmet": "no_helmet",
    "goggles": "no_goggle",
    "gloves": "no_gloves",
    "boots": "no_boots",
}
