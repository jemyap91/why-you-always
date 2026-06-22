# Dish Counter — Design Spec

**Date:** 2026-06-22
**Status:** Approved (design); pending implementation plan

## Purpose

A local computer-vision app that keeps a fair, running tally of how many dishes
each of two people (referred to here as **You** and **Wife**) washes at a shared
kitchen sink, captured by a Logitech Brio 100 webcam. The goal is a trustworthy,
hands-free scoreboard — accuracy of *attribution* matters more than catching
every single dish.

## Core decisions (locked during brainstorming)

1. **Attribution = hand skin tone.** The two people have visibly different skin
   tones. After MediaPipe locates a hand, we sample its region color and classify
   it against a calibrated per-person skin-tone profile. No face recognition, no
   stored images.
2. **Counting trigger = hand-off / removal from sink.** A wash is counted when a
   tracked hand moves from the **sink zone** into the **drying zone**. This
   avoids dish-object classification entirely (COCO detectors lack a "plate"
   class and suds/occlusion make object counting unreliable). We track a labeled
   hand crossing a boundary, not ceramics.
3. **Detector = MediaPipe Hands first**, placed behind a `HandDetector` Protocol
   so a HuggingFace hand/person detector can be swapped in later with a one-file
   change. (Python 3.14 in the environment does not support MediaPipe; the app
   pins a Python 3.12 virtualenv.)
4. **Interface = live local web dashboard** (FastAPI + WebSocket): live annotated
   feed, zone overlays, and a You-vs-Wife scoreboard (today + all-time).
5. **Bias toward undercounting.** When identity confidence is low, the event is
   tagged `uncertain` and **not** counted — we never mis-attribute, to keep the
   scoreboard trustworthy.
6. **Privacy:** store small skin-tone color profiles only, never raw images of
   faces or hands.

## Architecture

Single Python application, two threads sharing state through a thread-safe store:

```
Capture+Vision thread ("engine"):
  Brio → OpenCV frame → HandDetector → HandTracker →
  IdentityClassifier → ZoneEventEngine → CountStore
        │ publishes (latest annotated frame + counts)
        ▼
Web thread (FastAPI + WebSocket):
  streams annotated frames + scoreboard to the browser
```

The engine never blocks on the web layer. The dashboard reads the latest
published state, so CV framerate is independent of connected browser tabs.

## Components

Each component is isolated, has a single responsibility, communicates through a
well-defined interface, and is testable without hardware.

| Component | Responsibility | Depends on |
|---|---|---|
| `Camera` | Open the Brio, yield frames; reconnect on failure | OpenCV |
| `HandDetector` (Protocol) | frame → list of hand boxes + landmarks | MediaPipe impl (swappable) |
| `HandTracker` | assign stable IDs across frames via IoU matching | — |
| `IdentityClassifier` | hand region → `You` / `Wife` / `uncertain` via skin-tone profile | calibration profiles |
| `ZoneEventEngine` | detect sink→drying crossings, debounce → emit `WashEvent` | zone config |
| `CountStore` | apply events; hold today + all-time totals | SQLite |
| `Calibrator` | capture skin-tone profiles + draw the two zones | — |
| `dashboard` | FastAPI app: video stream + live scoreboard | engine state |
| `config` | Pydantic settings (zones, thresholds, camera index) | YAML/env |

### Shared domain types

- `Hand` dataclass: `id`, `bbox`, `landmarks`, `region_pixels`, `confidence`.
  `IdentityClassifier` and `ZoneEventEngine` only ever see `Hand` — never
  MediaPipe types. This dependency inversion is what makes the detector swap a
  one-file change.
- `WashEvent` dataclass: `person`, `timestamp`, `confidence`, `hand_id`.

## Data flow & the counting rule

A `WashEvent(person, timestamp)` fires when a tracked hand:

1. was present in the **sink zone** within the last `presence_window` seconds, and
2. its centroid then enters the **drying zone**, and
3. that hand-ID has not fired within `cooldown` seconds (debounce against jitter
   and boundary lingering).

Identity is **locked from the sink-zone phase** (the most reliable color sample,
hand fully wet/visible), not at the moment of crossing. If the locked identity is
`uncertain`, the event is recorded but flagged and excluded from counts.

## Calibration (one-time, re-runnable)

A `calibrate` command:

1. Each person holds a hand in the frame; capture the median skin chroma in
   **YCrCb** (chroma channels are robust to brightness changes) → store as that
   person's profile.
2. User drags two rectangles defining the **sink zone** and **drying zone**.
3. All saved to `config.yaml`. No raw images persisted.

## Persistence

- SQLite, append-only `events` table is the source of truth
  (`person, timestamp, confidence, counted`).
- Counts are **derived** from the event log, so totals are always reconstructable
  and a bad frame can never corrupt cumulative state.
- Dashboard shows **today** and **all-time** totals; full history is queryable.

## Error handling

- **Camera disconnect:** engine retries with exponential backoff; dashboard shows
  "camera offline".
- **0 / 1 / many hands per frame:** all valid; no hands = idle.
- **Low-confidence identity:** event tagged `uncertain`, not counted.
- **Missing config:** app refuses to start with a clear "run `calibrate` first"
  message.

## Testing strategy

- **Unit, no camera:** `HandTracker`, `IdentityClassifier`, `ZoneEventEngine`,
  `CountStore` tested with synthetic `Hand` objects / fake frames — deterministic.
- **Counting-rule suite (the heart of the app):** scripted hand trajectories
  (sink→drying, lingering on boundary, empty-handed exits, both hands at once)
  asserting exact counts.
- **Fake `Camera` and `HandDetector`** implementations run the whole pipeline in
  tests with no hardware.
- Components built test-first (TDD).

## Stack & project layout

- Python **3.12** (venv; MediaPipe lacks 3.14 support).
- `opencv-python`, `mediapipe`, `fastapi` + `uvicorn`, `pydantic`,
  `sqlite3` (stdlib), `pytest`, `ruff`.
- Installable package: `pyproject.toml`, `src/` layout.
- CLI entry points: `dishcounter calibrate`, `dishcounter run`.

## Out of scope (YAGNI)

- More than two people.
- Face recognition / identity beyond the two calibrated hands.
- Dish-type classification (plate vs. bowl).
- Cloud sync, mobile app, notifications.
- Distinguishing washing from rinsing/drying gestures beyond the zone crossing.
