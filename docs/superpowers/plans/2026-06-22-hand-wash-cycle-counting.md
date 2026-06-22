# Hand Wash-Cycle Counting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count a wash from hand activity — a hand that dwells in the sink for `min_wash` seconds then leaves — instead of from unreliable YOLO dish detection, and remove the dish-detection subsystem entirely.

**Architecture:** MediaPipe hands → IoU tracker (with coasting to bridge flicker) feed both gesture/session identity and a new `WashCycleEngine` that fires one `WashEvent` per enter→dwell→leave cycle for the active washer. Dish detection (YOLO), the `Dish` type, and `ultralytics` are deleted.

**Tech Stack:** Python 3.12, MediaPipe (hands), OpenCV, Pydantic, pytest. (Ultralytics removed.)

## Global Constraints

- Python `>=3.12,<3.13`; ruff `line-length = 100`.
- All detection behind dependency-injected fakes; **no models load in tests**.
- Run tests with `.venv/bin/python -m pytest` and lint with `.venv/bin/ruff check .` (NOT bare `python`).
- TDD: failing test first, watch it fail, minimal implementation, watch it pass, commit.
- Identity is the gesture session's active washer; no active session → nothing counts.
- `WashEvent.confidence = 1.0` for a counted wash (deliberate, not probabilistic).

---

### Task 1: `WashCycleEngine` — count enter→dwell→leave cycles

**Files:**
- Create: `src/dishcounter/washcycle.py`
- Test: `tests/test_washcycle.py`

**Interfaces:**
- Consumes: `Zone` (`.contains(point)`), `Hand` (`.id`, `.centroid`), `WashEvent`, and (in one integration test) `IouTracker`.
- Produces: `WashCycleEngine(sink, min_wash=3.0, cooldown=3.0)` with
  `process(hands, now, active_washer) -> list[WashEvent]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_washcycle.py`:

```python
from dishcounter.config import Zone
from dishcounter.domain import Hand
from dishcounter.tracker import IouTracker
from dishcounter.washcycle import WashCycleEngine

SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _hand(cx, cy, hid):
    return Hand(id=hid, bbox=(cx - 10, cy - 10, cx + 10, cy + 10))


def _in(hid):       # hand in the sink
    return _hand(50, 50, hid)


def _out(hid):      # hand present but outside the sink
    return _hand(50, 200, hid)


def test_dwell_then_leaving_sink_counts_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    ev = eng.process([_out(1)], 3.5, "You")
    assert len(ev) == 1 and ev[0].person == "You" and ev[0].source_id == 1
    assert ev[0].confidence == 1.0


def test_dwell_too_short_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 2.0, "You")
    assert eng.process([_out(1)], 2.5, "You") == []


def test_hand_disappearing_after_dwell_counts():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    assert len(eng.process([], 3.5, "You")) == 1


def test_two_hands_leaving_together_collapse_to_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1), _hand(60, 50, 2)], 0.0, "You")
    eng.process([_in(1), _hand(60, 50, 2)], 3.0, "You")
    assert len(eng.process([], 3.5, "You")) == 1


def test_no_session_counts_nothing():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, None)
    eng.process([_in(1)], 3.0, None)
    assert eng.process([], 3.5, None) == []


def test_re_entering_sink_counts_again_after_cooldown():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    eng.process([], 3.5, "You")                 # first count
    eng.process([_in(1)], 8.0, "You")           # new visit
    eng.process([_in(1)], 11.0, "You")
    assert len(eng.process([], 11.5, "You")) == 1


def test_hand_never_leaving_sink_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 5.0, "You")
    assert eng.process([_in(1)], 10.0, "You") == []


def test_hand_flicker_during_wash_counts_once():
    # Regression for the old overcounting: detection flicker while washing must
    # not spawn multiple counts — tracker coasting keeps one stable hand id.
    tracker = IouTracker(iou_threshold=0.3, coast_seconds=2.0)
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    seq = [(0.0, True), (1.8, False), (2.0, True), (3.8, False),
           (4.0, True), (8.0, False), (10.0, False)]
    count = 0
    for now, present in seq:
        hands = tracker.update([Hand(id=None, bbox=(40, 40, 60, 60))] if present else [], now)
        count += len(eng.process(hands, now, "You"))
    assert count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_washcycle.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dishcounter.washcycle'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/dishcounter/washcycle.py`:

```python
"""Counting rule: one wash = a hand dwells in the sink for >= min_wash seconds,
then leaves it (out of the zone, or out of frame). Credited to the active
session washer; a per-person cooldown collapses two-handed exits."""

from __future__ import annotations

from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Hand, WashEvent


@dataclass
class _Visit:
    enter_time: float | None = None  # start of the current in-sink visit
    last_in_sink: float = 0.0        # last time the hand was seen in the sink


class WashCycleEngine:
    def __init__(self, sink: Zone, min_wash: float = 3.0, cooldown: float = 3.0) -> None:
        self._sink = sink
        self._min_wash = min_wash
        self._cooldown = cooldown
        self._visits: dict[int, _Visit] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self, hands: list[Hand], now: float, active_washer: str | None
    ) -> list[WashEvent]:
        present: set[int] = set()
        events: list[WashEvent] = []
        for hand in hands:
            if hand.id is None:
                continue
            present.add(hand.id)
            visit = self._visits.setdefault(hand.id, _Visit())
            if self._sink.contains(hand.centroid):
                if visit.enter_time is None:
                    visit.enter_time = now
                visit.last_in_sink = now
            else:
                self._close(visit, hand.id, now, active_washer, events)

        for hid in list(self._visits):
            if hid in present:
                continue
            self._close(self._visits[hid], hid, now, active_washer, events)
            del self._visits[hid]
        return events

    def _close(self, visit, hand_id, now, active_washer, events) -> None:
        if visit.enter_time is None:
            return
        dwell = visit.last_in_sink - visit.enter_time
        visit.enter_time = None  # close the visit regardless of outcome
        if dwell < self._min_wash or active_washer not in ("You", "Wife"):
            return
        last = self._last_person_fire.get(active_washer)
        if last is not None and (now - last) < self._cooldown:
            return
        events.append(
            WashEvent(person=active_washer, timestamp=now, confidence=1.0,
                      source_id=hand_id)
        )
        self._last_person_fire[active_washer] = now
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_washcycle.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/washcycle.py tests/test_washcycle.py
git commit -m "feat: add WashCycleEngine counting hand dwell-and-leave cycles"
```

---

### Task 2: Cutover — Engine counts wash cycles; drop dish detection from the pipeline

**Files:**
- Modify: `src/dishcounter/config.py` (add `min_wash`, additive)
- Modify: `src/dishcounter/engine.py`
- Modify: `src/dishcounter/cli.py:60-70`
- Test: `tests/test_engine_e2e.py` (rewrite), `tests/test_annotate.py` (rewrite)

**Interfaces:**
- Consumes: `WashCycleEngine` (Task 1), `IouTracker`, `recognize_gesture`, `SessionController`.
- Produces: `Engine(camera, detector, config, store, state, clock=..., jpeg_encoder=..., annotator=...)` (no `dish_detector`); `annotate(frame, config, hands, active_washer, gesture)`; `Thresholds.min_wash: float = 3.0`.

This task is additive on config and leaves `dish_detector.py`, `fusion.py`, the
`Dish` type, dish config fields, and `ultralytics` in place (removed in Task 3)
so the suite stays green.

- [ ] **Step 1: Add `min_wash` to config (additive)**

In `src/dishcounter/config.py`, add one field to `Thresholds` (keep all existing
fields, including `exit_grace` and the dish settings, for now):

```python
class Thresholds(BaseModel):
    exit_grace: float = 1.5          # (unused after wash-cycle cutover; removed in cleanup)
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act
    track_coast: float = 2.0         # seconds a lost hand track is kept alive
    min_wash: float = 3.0            # seconds a hand must dwell in the sink to count
```

- [ ] **Step 2: Write the failing tests (rewrite e2e + annotate)**

Replace the entire contents of `tests/test_engine_e2e.py`:

```python
from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.domain import Hand
from dishcounter.engine import Engine
from dishcounter.state import SharedState
from dishcounter.store import CountStore

_FINGERS = {"index": (8, 6, 0.40), "middle": (12, 10, 0.48),
            "ring": (16, 14, 0.56), "pinky": (20, 18, 0.64)}


def _landmarks(extended):
    pts = [(0.5, 0.9)] * 21
    for name, (tip, pip, x) in _FINGERS.items():
        pts[pip] = (x, 0.5)
        pts[tip] = (x, 0.25 if name in extended else 0.55)
    return pts


def _hand(cx, cy, extended):
    return Hand(id=None, bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
                landmarks=_landmarks(extended))


def _fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def _engine(cfg, hand_script, frames, times):
    return Engine(
        FakeCamera(frames),
        FakeHandDetector(hand_script),
        cfg,
        CountStore(":memory:"),
        SharedState(),
        clock=_fake_clock(times),
        jpeg_encoder=lambda frame: b"jpeg",
    )


def test_gesture_session_then_wash_counts_one_for_you(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})                          # gesture, outside sink
    wash = _hand(50, 50, {"index", "middle", "ring"})         # 'other', in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
    )
    engine.run()  # FakeCamera exhausts after 5 frames
    assert engine._store.totals(8.0)["all_time"] == {"You": 1, "Wife": 0}


def test_wash_without_a_session_counts_nothing(sample_config, blank_frame):
    wash = _hand(50, 50, {"index", "middle", "ring"})
    engine = _engine(
        sample_config,
        hand_script=[[wash], [wash], []],
        frames=[blank_frame] * 3,
        times=[0.0, 3.5, 6.0],
    )
    engine.run()
    assert engine._store.totals(6.0)["all_time"] == {"You": 0, "Wife": 0}
```

Replace the entire contents of `tests/test_annotate.py`:

```python
import numpy as np

from dishcounter.config import Config, Zone
from dishcounter.domain import Hand
from dishcounter.engine import annotate


def _cfg() -> Config:
    return Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=50, y2=50))


def test_annotate_runs_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]  # centroid (20,20) -> in sink
    out = annotate(frame, _cfg(), hands, "You", "one")
    assert out.shape == frame.shape
    assert out.any()


def test_annotate_runs_with_no_session_or_hands():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [], None, "other")
    assert out.shape == frame.shape
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_engine_e2e.py tests/test_annotate.py -q`
Expected: FAIL — `Engine.__init__` still requires `dish_detector`; `annotate` arity mismatch.

- [ ] **Step 4: Rewrite the Engine (`src/dishcounter/engine.py`)**

Replace the imports block (drop `DishDetector`, `FusionEngine`; add
`WashCycleEngine`):

```python
from dishcounter.config import Config
from dishcounter.domain import WashEvent
from dishcounter.gesture import recognize_gesture
from dishcounter.session import SessionController
from dishcounter.state import SharedState
from dishcounter.store import CountStore
from dishcounter.tracker import IouTracker
from dishcounter.washcycle import WashCycleEngine
```

Replace `annotate` with:

```python
def annotate(frame: np.ndarray, config: Config, hands, active_washer, gesture
             ) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    z = config.sink_zone
    cv2.rectangle(out, (z.x1, z.y1), (z.x2, z.y2), (255, 0, 0), 2)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        color = (0, 255, 0) if z.contains(hand.centroid) else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if active_washer:
        banner, color = f"Session: {active_washer}", (0, 255, 0)
    else:
        banner = "Session: none - show 1 finger (You) / 2 (Wife)"
        color = (0, 165, 255)
    cv2.putText(out, banner, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"gesture: {gesture}", (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
```

In `Engine.__init__`, remove the `dish_detector` parameter and its assignment,
and replace the tracker/session/engine block:

```python
    def __init__(
        self,
        camera,
        detector,
        config: Config,
        store: CountStore,
        state: SharedState,
        clock: Callable[[], float] = time.time,
        jpeg_encoder: Callable[[np.ndarray], bytes] = encode_jpeg,
        annotator: Callable = annotate,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._running = False

        t = config.thresholds
        self._hand_tracker = IouTracker(
            iou_threshold=t.iou_match, coast_seconds=t.track_coast
        )
        self._session = SessionController(hold_seconds=t.gesture_hold)
        self._counter = WashCycleEngine(
            config.sink_zone, min_wash=t.min_wash, cooldown=t.cooldown
        )
```

Replace `process_frame` (keep the existing `_resolve_gesture` staticmethod as-is):

```python
    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame), now)
        gesture = self._resolve_gesture(hands)
        active = self._session.update(gesture, now)
        events = self._counter.process(hands, now, active)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands, active, gesture)
        self._state.publish(
            self._encode(annotated), self._store.totals(now), camera_online=True
        )
        return events
```

- [ ] **Step 5: Update `cli.py`**

In `src/dishcounter/cli.py`, delete the
`from dishcounter.dish_detector import YoloWorldDishDetector` line and construct
the engine without a dish detector:

```python
    state = SharedState()
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        config,
        CountStore(db),
        state,
    )
```

- [ ] **Step 6: Run the full suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (e2e + annotate rewritten; `test_fusion.py`, `test_dish_detector.py`,
`test_overcounting.py` still pass — those files are removed in Task 3).

- [ ] **Step 7: Commit**

```bash
git add src/dishcounter/config.py src/dishcounter/engine.py src/dishcounter/cli.py \
        tests/test_engine_e2e.py tests/test_annotate.py
git commit -m "feat: count wash cycles from hands; drop dish detector from pipeline"
```

---

### Task 3: Cleanup — delete the dish-detection subsystem

**Files:**
- Delete: `src/dishcounter/dish_detector.py`, `tests/test_dish_detector.py`,
  `src/dishcounter/fusion.py`, `tests/test_fusion.py`, `tests/test_overcounting.py`
- Modify: `src/dishcounter/domain.py` (remove `Dish`)
- Modify: `src/dishcounter/config.py` (remove dish fields, `DEFAULT_DISH_CLASSES`, `exit_grace`)
- Modify: `tests/test_config.py`, `tests/conftest.py`
- Modify: `pyproject.toml`, `requirements.txt` (remove `ultralytics`)

**Interfaces:**
- Produces: `Config` with no dish settings; `Thresholds` without `exit_grace`;
  no `Dish` type.

- [ ] **Step 1: Write/adjust the failing test**

In `tests/test_config.py`, add (and update any test that still passes dish
fields or asserts `exit_grace`):

```python
def test_config_has_no_dish_settings_and_no_exit_grace():
    from dishcounter.config import Config, Zone

    cfg = Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=10, y2=10))
    assert not hasattr(cfg, "dish_classes")
    assert not hasattr(cfg, "dish_conf")
    assert not hasattr(cfg, "yolo_model")
    assert not hasattr(cfg.thresholds, "exit_grace")
    assert cfg.thresholds.min_wash == 3.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -q`
Expected: FAIL — dish fields / `exit_grace` still present.

- [ ] **Step 3: Remove the dish subsystem**

Delete the files:

```bash
git rm src/dishcounter/dish_detector.py tests/test_dish_detector.py \
       src/dishcounter/fusion.py tests/test_fusion.py tests/test_overcounting.py
```

In `src/dishcounter/domain.py`, delete the `Dish` dataclass (keep `Hand`,
`WashEvent`). If `numpy` is then unused in `domain.py`, remove its import; let
`ruff check .` decide.

In `src/dishcounter/config.py`, remove the `DEFAULT_DISH_CLASSES` constant,
remove `exit_grace` from `Thresholds`, and remove `dish_classes`/`dish_conf`/
`yolo_model` from `Config`. Resulting shapes:

```python
class Thresholds(BaseModel):
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act
    track_coast: float = 2.0         # seconds a lost hand track is kept alive
    min_wash: float = 3.0            # seconds a hand must dwell in the sink to count


class Config(BaseModel):
    camera_index: int
    sink_zone: Zone
    thresholds: Thresholds = Thresholds()
```

(Keep the existing `load`/`save` methods.)

In `tests/conftest.py`, confirm `sample_config` still constructs (it already
omits dish fields and uses `Thresholds(gesture_hold=1.0)` — no change needed
unless it references removed fields).

- [ ] **Step 4: Remove the dependency**

In `pyproject.toml`, delete the `"ultralytics>=8.2",` line from `dependencies`.
In `requirements.txt`, delete the `ultralytics>=8.2` line.

- [ ] **Step 5: Run the full suite + lint + grep**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

Run: `.venv/bin/ruff check .`
Expected: `All checks passed!`

Run: `grep -rn "Dish\b\|dish_detector\|FusionEngine\|ultralytics\|exit_grace\|dish_classes\|yolo_model" src/ tests/`
Expected: no matches (the word "dish" in comments/strings is fine; this targets the symbols).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove YOLO dish-detection subsystem and ultralytics dep"
```

(Stage with `git add -A` only after `git status` confirms nothing unrelated is
present.)

---

### Task 4: Update the README for wash-cycle counting

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Rewrite the affected sections**

Update `README.md` to describe wash-cycle counting:
- Intro bullets: a wash is counted when your hand dwells in the sink then leaves
  (carrying the dish away); identity by gesture (1=You, 2=Wife, fist=end);
  nothing counts with no session. Remove the "dishes detected directly / YOLO"
  bullet.
- Section 2 (Setup): drop "Ultralytics" from what gets installed.
- Section 4 (Run): remove the YOLO weights first-run download note; the live
  feed shows hand boxes (green when in the sink) and the session banner; explain
  the workflow — start a session (1/2 fingers), wash a dish (hand in the sink a
  few seconds) then carry it out, repeat; fist to end.
- Section 6 (Tuning): replace the dish-detection table and `exit_grace`/`track_coast`
  framing — list `min_wash` (seconds a hand must dwell to count), `cooldown`,
  `track_coast` (bridges hand-detection flicker), `gesture_hold`, `iou_match`.
  Remove `dish_classes`/`dish_conf`/`yolo_model`.
- Section 7 (How it works): update the pipeline diagram to a single hand
  detector feeding gesture/session and the `WashCycleEngine`; no YOLO.
- Section 8 (Troubleshooting): replace dish-detection rows with wash-cycle ones
  (e.g. "washes not counted → make the sink zone snug and dwell a couple seconds
  before lifting the dish out"; "overcounting → raise `min_wash`").
- Privacy: unchanged (sink rectangle + settings only).

- [ ] **Step 2: Verify the suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document hand wash-cycle counting"
```

---

## Self-Review

**Spec coverage:**
- WashCycleEngine enter→dwell→leave rule, min_wash gate, per-person cooldown, no-session-no-count, bounded state → Task 1.
- Hand-flicker regression (coasting prevents overcount) → Task 1.
- Engine wiring (single hand detector, coasting tracker, WashCycleEngine), overlay (in-sink tint + banner + gesture), cli → Task 2.
- `min_wash` config (add) / dish settings + `exit_grace` (remove) → Task 2 (add) + Task 3 (remove).
- Remove dish_detector, fusion, Dish, ultralytics → Task 3.
- README → Task 4.
- All detection behind fakes; no models in tests → Tasks 1, 2.

**Placeholder scan:** none — every code/test step has complete code; commands have expected output.

**Type consistency:** `WashCycleEngine(sink, min_wash, cooldown).process(hands, now, active_washer)`, `Engine(camera, detector, config, store, state, ...)`, `annotate(frame, config, hands, active_washer, gesture)`, and `Thresholds.min_wash` are used identically across the tasks that define and consume them. Task 2 keeps the dish config/files so the suite stays green; Task 3 removes them with their last references.
