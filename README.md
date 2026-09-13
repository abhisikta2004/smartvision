# SmartVision

Real-time PPE compliance and worker safety monitoring, trained on the [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe) detection dataset.

SmartVision finds people and safety gear in still images, uploaded video, and a live webcam. It raises visual and audio alerts when missing PPE is detected (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`) and when a person is in frame without a vest or helmet.

## Classes

Worn PPE: `helmet`, `gloves`, `vest`, `boots`, `goggles`  
Missing PPE: `no_helmet`, `no_goggle`, `no_gloves`, `no_boots`  
Other: `Person`, `none`

The dataset has no dedicated missing-vest label. SmartVision adds a site heuristic for that case.

## Setup

Python 3.11+ recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

The first run downloads Construction-PPE (~178 MB) via Ultralytics.

```bash
python scripts/train.py --epochs 30 --batch 8
```

On Apple Silicon, Ultralytics will use MPS when available. Weights are copied to `models/best.pt`.

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

1. **Live camera** — stream frames from the webcam and overlay detections.
2. **Image inspect** — upload a photo and download the annotated result.
3. **Video review** — scan a clip, jump through a violation timeline, download the annotated video.

Adjust confidence and optional site heuristics from the top bar. Alert sound can be muted.

## License

Construction-PPE is released under AGPL-3.0 by Ultralytics. If you use the dataset, cite:

Dalvi, M., Singh, N., Bhingarde, S., & Chalke, K. (2025). *Construction-PPE: Personal Protective Equipment Detection Dataset*. Ultralytics. https://docs.ultralytics.com/datasets/detect/construction-ppe
