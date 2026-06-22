# Hand wash-cycle counting — design

**Date:** 2026-06-22
**Status:** Approved (pre-implementation)

## Problem

Counting dishes from a YOLO object detector's appear/disappear events is
fundamentally unreliable in a real wet, cluttered kitchen. Detection flickers
on wet/suds/hand-occluded dishes and on background clutter; one physical dish
churns through many tracks. Evidence from live runs: counts fired continuously
(every ~3–5s) regardless of how many dishes were actually washed — first pinned
at the cooldown (3.0s), then ~5s after adding tracker coasting. Coasting only
slowed the churn; the occlusion gaps during scrubbing exceed any safe coast
window. The mechanism is wrong, not the tuning.

## Goal

Count a wash from **hand activity**, which MediaPipe tracks far more stably than
YOLO tracks dishes:

> One wash = a hand **dwells in the sink** for at least `min_wash` seconds, then
> **leaves the sink** (moves out of the sink zone, or out of frame to carry the
> dish away).

Each cycle counts one dish for the active session washer (set by gesture:
1 finger = You, 2 = Wife), with a per-person cooldown to collapse two-handed
exits. With no active session, nothing counts.

This **removes the entire YOLO dish-detection subsystem.**

## Non-goals

- Exact dish counting when several dishes are washed without the hand leaving
  the sink (e.g. rinsing a stack) — that counts once. The intended workflow is
  "wash one, carry it out," which is ~1:1.
- Object/dish detection of any kind (removed).
- Distinguishing simultaneous washers (turn-based; session gesture handles it).

## Approach

Reuse the stable, working pieces — MediaPipe hand detection, the IoU tracker
(now with coasting bridging hand flicker), the sink zone, and the gesture-driven
session. Replace the dish `FusionEngine` with a `WashCycleEngine` that watches
each hand's dwell in the sink and fires on exit. Drop YOLO, the `Dish` type, and
all dish config/dependencies.

### Pipeline

```
Camera ─ HandDetector (MediaPipe) ─ IouTracker(coast) ─┬─ recognize_gesture ─ SessionController ─┐
                                                       └─ WashCycleEngine ──────────────────────┴─► CountStore
                                                          (active washer in)
```

## Components

### `WashCycleEngine` (new module `washcycle.py`, replaces `fusion.py`)
- `WashCycleEngine(sink: Zone, min_wash: float = 3.0, cooldown: float = 3.0)`.
- `process(hands: list[Hand], now: float, active_washer: str | None) -> list[WashEvent]`.
- Per hand-id state: `enter_time: float | None` (start of the current in-sink
  visit, else None), `last_in_sink: float` (time the hand was last seen in the
  sink during this visit).
- Each frame:
  1. For each present hand (id not None): if its centroid is in `sink_zone`,
     start a visit (`enter_time = now` if not already in one) and stamp
     `last_in_sink = now`. If the hand is present but **outside** the sink, close
     any open visit (see *fire rule*).
  2. For each tracked hand id **not** present this frame (disappeared — i.e.
     truly gone after the tracker's coast window): close its visit, then drop the
     state.
- **Fire rule (closing a visit):** if `enter_time is not None`, compute
  `dwell = last_in_sink - enter_time` and set `enter_time = None`. Emit one
  `WashEvent(person=active_washer, timestamp=now, confidence=1.0, source_id=hand_id)`
  iff `dwell >= min_wash` **and** `active_washer in ("You", "Wife")` **and** the
  per-person cooldown has elapsed (`now - last_fire[active_washer] >= cooldown`).
  No active session → visit closes silently, nothing emitted.

This makes a wash a deliberate enter→dwell→leave cycle. Reaching into the sink
briefly (`dwell < min_wash`) never counts. Two hands leaving together collapse to
one via the cooldown.

### Tracker coasting on hands (`engine.py` wiring)
The hand tracker is constructed with `coast_seconds=track_coast` so MediaPipe
flicker doesn't fragment a dwell or cause a spurious exit. A hand carried out of
frame is coasted (re-emitted in its last in-sink position) for `track_coast`
seconds, then dropped — at which point its visit closes and fires. (The
`IouTracker` coasting added previously is reused as-is; only the wiring changes
from the dish tracker to the hand tracker.)

### `engine.py` (changed)
- `Engine(camera, detector, config, store, state, clock=..., jpeg_encoder=..., annotator=...)`
  — the `dish_detector` parameter is removed.
- Builds one `IouTracker(iou_threshold=t.iou_match, coast_seconds=t.track_coast)`
  for hands, a `SessionController(hold_seconds=t.gesture_hold)`, and a
  `WashCycleEngine(config.sink_zone, min_wash=t.min_wash, cooldown=t.cooldown)`.
- `process_frame`: detect+track hands (passing `now`), resolve one gesture,
  update the session, call `WashCycleEngine.process(hands, now, active)`, record
  events, annotate, publish.
- `annotate(frame, config, hands, active_washer, gesture)` — draws the sink zone
  (blue), each hand box (tinted **green when its centroid is in the sink**, amber
  otherwise) so you can see washing register, the session banner
  (`Session: You` / `Wife` / `none - show 1 finger (You) / 2 (Wife)`), and the
  `gesture:` label. No dish drawing.

### `config.py` (changed)
- `Thresholds`: remove `exit_grace`; add `min_wash: float = 3.0`. Keep
  `cooldown`, `iou_match`, `gesture_hold`, `track_coast`.
- `Config`: remove `dish_classes`, `dish_conf`, `yolo_model` (and the
  `DEFAULT_DISH_CLASSES` constant). Keep `camera_index`, `sink_zone`,
  `thresholds`.

### Removed
- `src/dishcounter/dish_detector.py` and `tests/test_dish_detector.py`.
- `src/dishcounter/fusion.py` and `tests/test_fusion.py`,
  `tests/test_overcounting.py` (dish-based) — replaced by `washcycle` tests.
- The `Dish` dataclass in `domain.py` (no longer used).
- `ultralytics` from `pyproject.toml` and `requirements.txt`.

### `cli.py` (changed)
- Drop the `YoloWorldDishDetector` import/construction; build
  `Engine(Camera(...), MediaPipeHandDetector(), config, store, state)`.

## Data flow / attribution

Identity is the session's `active_washer` at the moment a visit closes (sessions
are stable across a wash, so this is correct). `WashEvent.source_id` carries the
hand id. `CountStore` is unchanged; only `You`/`Wife` events have `counted=1`.

## Error handling / edge cases

- **No landmarks / no hands** (dish-free frames, fakes in tests): no visits, no
  events.
- **Hand flicker mid-wash:** bridged by tracker coasting, so the dwell stays
  continuous.
- **Hand leaves sink but stays in frame:** visit closes immediately on the
  out-of-sink frame (no coast delay needed).
- **Hand carried out of frame:** coasted for `track_coast`, then dropped → visit
  closes and fires.
- **Two hands:** per-person cooldown collapses simultaneous exits to one count.
- **Reaching for sponge / brief dip:** `dwell < min_wash` → no count.
- **Hand never leaves the sink (washes a stack):** counts once (documented
  limitation).
- Per-hand state is bounded: dropped when a hand disappears.

## Testing

All detection behind dependency-injected fakes; no models load in tests.

- **WashCycleEngine:** dwell ≥ `min_wash` then hand moves out of sink → one
  event for the active washer; dwell < `min_wash` → no event; hand disappears
  after a qualifying dwell → one event; two hands leaving together → collapse to
  one; no active session → no event; re-entering the sink after a count starts a
  new countable visit; a hand that stays in the sink the whole time → no event.
- **Hand-flicker regression** (tracker + engine): a hand whose detection
  flickers while dwelling in the sink, then leaves, counts **once** (proves
  coasting prevents the old overcounting failure mode).
- **engine e2e:** scripted hand fakes drive a full gesture session: show
  `one` → hand dwells in sink → hand leaves → one count for You; with no session
  the same motion counts nothing.
- **config/calibrate:** config has `min_wash`, no dish settings/`exit_grace`;
  calibrate unchanged (sink zone only).

## Out of scope / future

- A live dwell-progress readout on the overlay (in-sink tint only for now).
- Counting multiple dishes washed in one continuous sink dwell.
- Reintroducing object detection.
