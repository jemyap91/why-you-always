# Dish-presence gate — design

**Date:** 2026-06-23
**Status:** Approved (pre-implementation)

## Problem

The current hand wash-cycle counting fires whenever a hand **dwells in the sink
for `min_wash` seconds and then leaves**. That trigger is stable, but it
overcounts: hand activity is not the same as washing a dish. Rinsing your hands,
wringing a sponge, or splashing water all dwell-and-leave and get counted,
inflating the score.

A previous attempt to count from a **dish object detector** (YOLO-World +
`FusionEngine`, counting each dish leaving the sink) was built and then removed.
Live-run evidence: detection flickered on wet/suds/hand-occluded dishes, one
physical dish churned through many tracks, and counts fired continuously
regardless of how many dishes were washed. The conclusion recorded at the time:
*the mechanism is wrong, not the tuning* — per-dish **tracking and counting** is
not reliable in a wet, cluttered sink.

## Goal

Cut the false counts **without** putting unreliable per-dish tracking back on the
counting path. Keep the stable hand dwell-and-leave cycle as the trigger, and use
the dish detector for one narrow, reliable job: confirm that **a dish was present
in the sink during the wash**. No dish seen during the dwell → it was hand/sponge
rinsing → not counted.

> One wash = a hand dwells in the sink ≥ `min_wash` seconds, **a dish is seen in
> the sink during that dwell**, then the hand leaves — credited to the active
> session washer.

The model performs a **presence check, not tracking/counting**. It never has to
hold a stable dish identity, so the flicker that killed the prior approach is
irrelevant: it only needs to fire on *some* frame during a multi-second scrub.

## Non-goals

- Counting individual dishes (no per-dish tracking; one dwell = one count,
  same as today — washing a stack in one continuous dwell still counts once).
- Reintroducing `FusionEngine` / dish-driven counting.
- Changing gesture sessions, calibration (sink zone only), storage, or the
  dashboard scoreboard.
- Training a custom model — we reuse off-the-shelf open-vocabulary YOLO-World.

## Approach

Reuse the proven pieces — MediaPipe hands, the IoU tracker with coasting, the
sink zone, gesture-driven sessions, and `WashCycleEngine`. Add the recovered
`DishDetector` (YOLO-World) as a **gate**: `WashCycleEngine` only fires a wash if
a dish was confirmed in the sink during the hand's dwell.

### Pipeline

```
Camera ─ MediaPipe(hands) ─ IouTracker(coast) ─┬─ recognize_gesture ─ SessionController ─┐
                                               │                                          │
                                               └─ WashCycleEngine ────────────────────────┤─► CountStore
                                                  ▲ dish_seen                              │
              DishDetector (YOLO-World, hand-gated) ─┘                                     │
              (publishes annotated frame + counts) ──────────────────────────────────────┘
```

### The gate rule

`WashCycleEngine` keeps its per-hand-id visit state and adds a `dish_hits`
counter per open in-sink visit.

Each frame, `process(hands, now, active_washer, dish_seen)`:

1. Hand dwell bookkeeping is unchanged: a hand whose centroid is in the sink
   starts/continues a visit and stamps `last_in_sink`; a present hand outside the
   sink, or a disappeared (post-coast) hand, **closes** its visit.
2. **New:** if `dish_seen` is True this frame, increment `dish_hits` for every
   hand that currently has an **open in-sink visit**.
3. **Fire rule (closing a visit):** emit one
   `WashEvent(person=active_washer, timestamp=now, confidence=1.0, source_id=hand_id)`
   iff **all** of:
   - `dwell = last_in_sink - enter_time >= min_wash` (unchanged), **and**
   - `dish_hits >= dish_min_hits` (**new**), **and**
   - `active_washer in ("You", "Wife")` (unchanged), **and**
   - per-person cooldown elapsed (unchanged).
   Otherwise the visit closes silently. `dish_hits` resets with the visit.

`dish_seen` means **"a dish-detector run completed this frame and found a
dish-class detection whose centroid lies in the sink zone."** It is True **at
most once per `dish_interval`** (the engine throttles detector runs), so
`dish_hits` counts *detector confirmations*, not frames. Default
`dish_min_hits = 1`: any single in-sink dish sighting during the dwell validates
the wash.

This biases toward **undercounting** — if the model misses a real plate, the
wash is dropped rather than a false count added — which is the safe direction
given the complaint is overcounting. Disable the gate by setting
`dish_min_hits: 0` (fires exactly like today).

### Compute model (hand-gated, throttled detection)

The dish detector is expensive (YOLO-World alongside MediaPipe is single-digit
FPS). It runs **only when a wash could be happening**:

- a session is active, **and**
- at least one tracked hand's centroid is currently in the sink zone, **and**
- at least `dish_interval` seconds have passed since the last dish-detector run.

Otherwise it does not run and `dish_seen` is False. During a multi-second dwell
this samples a few times — enough to establish presence — and YOLO-World is idle
whenever nobody is actively washing. "Dish in sink" reuses `Zone.contains` on the
dish centroid.

## Components

### Recovered from git (near-verbatim)

- **`domain.py`** — re-add the `Dish` dataclass
  (`id: int | None`, `bbox: (x1,y1,x2,y2)`, `label: str = ""`,
  `confidence: float = 0.0`, `centroid` property). `Hand`/`WashEvent` unchanged.
- **`dish_detector.py`** (+ `tests/test_dish_detector.py`) — recovered from commit
  `ce9c090`:
  - `DishDetector` Protocol (`runtime_checkable`), `detect(frame) -> list[Dish]`.
  - `FakeDishDetector(script)` — deterministic, for tests/headless.
  - `dishes_from_detections(detections, min_conf) -> list[Dish]` — pure mapping +
    confidence filter (unit-tested).
  - `YoloWorldDishDetector(classes, conf, model="yolov8s-worldv2.pt")` — lazy-imports
    `ultralytics`; `# pragma: no cover` on the model-loading/IO paths.
- **`ultralytics`** — re-added to `pyproject.toml` and `requirements.txt`.

### Changed

- **`washcycle.py`** — `WashCycleEngine(sink, min_wash=3.0, cooldown=3.0, dish_min_hits=1)`;
  `process(hands, now, active_washer, dish_seen)` per the gate rule above.
- **`engine.py`** — constructor takes an injected `dish_detector`. `process_frame`:
  detect+track hands → resolve gesture → update session → decide whether to run
  the dish detector (hand-gated + throttled, tracking `last_dish_run`) → compute
  `dish_seen` → `washcycle.process(hands, now, active, dish_seen)` → record →
  annotate → publish. A dish-detector exception on a frame is caught →
  `dish_seen=False`, loop continues. `annotate(...)` additionally draws dish boxes
  when a detection occurred and a small "dish ✓" marker while a gated wash is in
  progress.
- **`config.py`** — `Config` gains `dish_classes: list[str]`
  (default `["plate","bowl","cup","glass","mug","fork","knife","spoon"]`),
  `dish_conf: float = 0.4`, `yolo_model: str = "yolov8s-worldv2.pt"`. `Thresholds`
  gains `dish_interval: float = 0.5` and `dish_min_hits: int = 1`. **All have
  defaults, so existing `config.yaml` files load unchanged.** `calibrate.py` is
  unchanged (sink zone only).
- **`cli.py`** — build
  `YoloWorldDishDetector(config.dish_classes, config.dish_conf, config.yolo_model)`
  and pass it into `Engine`. README: weights download (~340MB, YOLO-World +
  CLIP text encoder) on first run.

## Data flow / attribution

Identity is the session's `active_washer` at the moment the visit closes
(unchanged). The dish detector contributes **only** the boolean `dish_seen` gate;
it never touches identity, tracking, or storage. `WashEvent` shape and
`CountStore` are unchanged; only `You`/`Wife` events have `counted=1`.

## Error handling / edge cases

- **YOLO-World load failure** surfaces at startup in `run` (cli constructs the
  detector), never mid-loop, never in tests.
- **Detector throws on a frame** → caught in the engine, `dish_seen=False` for
  that frame, loop continues (a transient detector error costs at most a missed
  confirmation, never a crash).
- **Hand-only motion (no dish):** `dish_hits` stays 0 → visit closes silently →
  no count. *(This is the fix.)*
- **Dish flicker during scrub:** one confirmation is enough (`dish_min_hits=1`);
  the throttled sampler only needs the dish on a single run.
- **Dwell too short** (`< min_wash`): no count regardless of `dish_seen`.
- **No active session:** detector does not run; nothing counts.
- **Two hands in the sink:** both open visits accrue `dish_hits`; the per-person
  cooldown collapses simultaneous exits to one count (unchanged).
- **Re-entry after a count:** new visit starts fresh `enter_time`/`dish_hits`.
- Per-hand state stays bounded: dropped when a hand disappears (unchanged).

## Testing

All detection behind dependency-injected fakes; **no ultralytics import in
tests**.

- **`dish_detector`** (recovered): `dishes_from_detections` confidence filtering;
  `FakeDishDetector` sequencing/last-repeat; satisfies the `runtime_checkable`
  Protocol.
- **`WashCycleEngine`** (gate): dwell ≥ `min_wash` **with** a `dish_seen`
  confirmation during the dwell → one event; the **same dwell with no
  `dish_seen`** → **no event** (core regression for hand-only false counts);
  `dish_seen` but `dwell < min_wash` → no event; re-entry after a count resets
  `dish_hits`; `dish_min_hits=0` disables the gate (fires like today); two hands
  leaving together → collapse to one via cooldown.
- **`engine` e2e:** scripted hand fakes + scripted `FakeDishDetector` through a
  gesture session — `show one` → hand dwells in sink while the fake reports a
  dish → one count for You; the **identical** hand motion with the fake reporting
  **no dishes** → **zero counts**; dish present but no session → zero.
- **Detector cadence:** a spy `FakeDishDetector` asserts the engine calls it only
  when a hand is in the sink **and** respects `dish_interval` (no call when no
  hand is in the sink, no call before the interval elapses).
- **config:** new fields default; an existing `config.yaml` (sink zone +
  current thresholds, no dish fields) still loads.

## Out of scope / future

- Per-dish counting / counting a washed stack as more than one.
- Cropping the frame to the sink ROI before YOLO for speed (optimization).
- A live dwell/confirmation progress readout on the overlay.
- Reintroducing dish-driven counting (`FusionEngine`).
