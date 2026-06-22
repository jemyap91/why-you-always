# Dish Counter

Local computer-vision scoreboard that tallies how many dishes **You** vs **Wife**
wash at a shared sink, captured by a Logitech Brio webcam. Attribution is by hand
skin tone; a wash is counted when a tracked hand moves from the sink zone into the
drying zone. Biased toward undercounting — uncertain events are never attributed.

## Requirements

- macOS with a webcam.
- Python 3.12 (MediaPipe has no 3.14 support): `brew install python@3.12`.

## Setup

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Use

```bash
.venv/bin/dishcounter calibrate   # capture both skin profiles, drag the two zones
.venv/bin/dishcounter run         # open http://127.0.0.1:8000
```

`calibrate` writes `config.yaml` (skin-tone profiles + zone coords only — no images).
`run` starts the vision engine and the dashboard; counts persist in `dishcounter.db`.

## Develop

```bash
.venv/bin/pytest        # full suite runs with zero hardware (fakes throughout)
.venv/bin/ruff check .
```

## How it works

`Camera → HandDetector → HandTracker → IdentityClassifier → ZoneEventEngine → CountStore`
runs on a vision thread and publishes the latest annotated frame + counts to a
thread-safe store; a FastAPI thread streams that to the browser. Counts are derived
from an append-only event log, so totals are always reconstructable.

## Privacy

Only small skin-tone color profiles and event rows are stored. No face/hand images
are ever persisted.
