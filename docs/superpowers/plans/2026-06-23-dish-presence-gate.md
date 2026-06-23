# Dish-presence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the current hand-dwell counter from logging false washes by gating each wash on a dish actually being present in the sink — using the recovered YOLO-World detector purely as a presence check, never as a counter.

**Architecture:** Keep the stable hand dwell-and-leave trigger (`WashCycleEngine`) and gesture sessions exactly as they are. Re-add the `Dish` type and the `DishDetector` (YOLO-World) from git. The Engine runs the dish detector only while a session is active and a hand is in the sink (throttled), and passes a per-frame `dish_seen` boolean into `WashCycleEngine`, which only fires a wash if a dish was confirmed during the hand's dwell.

**Tech Stack:** Python 3.12, MediaPipe (hands), Ultralytics YOLO-World (dishes), OpenCV, FastAPI, Pydantic, SQLite, pytest, ruff.

## Global Constraints

- Run all commands via the project venv: `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/python` — never bare `pytest`/`python`.
- The dish model does a **presence check, not tracking/counting**. It must never be put on the counting path; `WashCycleEngine` remains the only thing that fires `WashEvent`s.
- **Gate rule:** a visit fires only if `dwell >= min_wash` AND `dish_hits >= dish_min_hits` AND `active_washer in ("You","Wife")` AND the per-person cooldown has elapsed.
- `dish_seen` means "a dish-detector run completed this frame and found a dish whose centroid is in the sink zone." It is True at most once per `dish_interval`, so `dish_hits` counts detector confirmations, not frames.
- Defaults: `dish_min_hits = 1`, `dish_interval = 0.5`, `dish_conf = 0.4`, `yolo_model = "yolov8s-worldv2.pt"`, `dish_classes = ["plate","bowl","cup","glass","mug","fork","knife","spoon"]`.
- Bias toward **undercounting**: when in doubt, do not count.
- **No `ultralytics`/model load in tests** — all detection behind dependency-injected fakes. YOLO-World lazy-imports `ultralytics` inside the wrapper; the wrapper's model-loading/IO paths carry `# pragma: no cover`.
- Config fields are added **with defaults** so existing `config.yaml` files still load. Calibration is unchanged (sink zone only).
- Identity comes from the session's `active_washer` (unchanged). Labels are exactly `"You"`/`"Wife"`/`"uncertain"`. `WashEvent.source_id` (not `hand_id`).
- End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- `git add` only the files each task names — never `git add -A`/`git add .` (there is gitignored scratch under `.superpowers/`).

---

## File Structure

```
src/dishcounter/
  domain.py         # + re-add Dish dataclass                          (Task 1)
  dish_detector.py  # NEW: DishDetector Protocol, Fake, YOLO-World      (Task 2)
  config.py         # + dish_classes/dish_conf/yolo_model, dish_*thresh (Task 3)
  washcycle.py      # + dish_min_hits gate, dish_hits, dish_seen arg    (Task 4)
  engine.py         # + injected dish_detector, throttled detection     (Task 5)
                    # + annotate dish indicator                         (Task 6)
  cli.py            # + build YoloWorldDishDetector in _serve           (Task 7)
pyproject.toml      # + ultralytics dependency                          (Task 7)
requirements.txt    # + ultralytics                                     (Task 7)
README.md           # + dish-gate explanation + weights download note   (Task 8)
tests/
  test_domain.py        # + Dish centroid                              (Task 1)
  test_dish_detector.py # NEW (recovered)                              (Task 2)
  test_config.py        # + new-field defaults / back-compat load      (Task 3)
  test_washcycle.py     # updated for the gate + new gate tests        (Task 4)
  test_engine_e2e.py    # + dish_detector param + gate/cadence tests    (Task 5)
  test_annotate.py      # + dish indicator                             (Task 6)
```

---

### Task 1: Re-add the `Dish` domain type

**Files:**
- Modify: `src/dishcounter/domain.py`
- Test: `tests/test_domain.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Dish(id: int | None, bbox: tuple[int,int,int,int], label: str = "", confidence: float = 0.0)` with a `centroid -> tuple[float, float]` property (mirrors `Hand`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_domain.py`:
```python
def test_dish_centroid_is_bbox_center():
    from dishcounter.domain import Dish

    d = Dish(id=None, bbox=(10, 20, 30, 60), label="plate", confidence=0.9)
    assert d.centroid == (20.0, 40.0)
    assert d.label == "plate"
    assert d.confidence == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_domain.py::test_dish_centroid_is_bbox_center -v`
Expected: FAIL with `ImportError: cannot import name 'Dish'`.

- [ ] **Step 3: Add the `Dish` dataclass**

Append to `src/dishcounter/domain.py` (after `WashEvent`):
```python
@dataclass
class Dish:
    """A detected dish for one frame, from the object detector."""

    id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str = ""
    confidence: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_domain.py -v`
Expected: PASS (all domain tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/domain.py tests/test_domain.py
git commit -m "feat: re-add Dish domain type for the presence gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Recover the `DishDetector` module

**Files:**
- Create: `src/dishcounter/dish_detector.py`
- Test: `tests/test_dish_detector.py`

**Interfaces:**
- Consumes: `Dish` (Task 1).
- Produces:
  - `DishDetector` — `runtime_checkable` Protocol with `detect(frame) -> list[Dish]`.
  - `FakeDishDetector(script: list[list[Dish]])` — replays scripted frames, clamps to the last entry; empty script → `[[]]`.
  - `dishes_from_detections(detections: list[tuple[tuple[int,int,int,int], str, float]], min_conf: float) -> list[Dish]` — pure mapping + confidence filter.
  - `YoloWorldDishDetector(classes: list[str], conf: float, model: str = "yolov8s-worldv2.pt")` — lazy-imports `ultralytics`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dish_detector.py`:
```python
import numpy as np

from dishcounter.dish_detector import (
    DishDetector,
    FakeDishDetector,
    dishes_from_detections,
)
from dishcounter.domain import Dish


def test_fake_dish_detector_replays_script_and_clamps_to_last():
    d = Dish(id=None, bbox=(0, 0, 10, 10), label="plate", confidence=0.9)
    det = FakeDishDetector([[d], []])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert det.detect(frame) == [d]
    assert det.detect(frame) == []
    assert det.detect(frame) == []  # clamps to last script entry
    assert isinstance(det, DishDetector)


def test_dishes_from_detections_filters_low_confidence_and_maps_fields():
    raw = [
        ((0, 0, 20, 20), "plate", 0.8),
        ((5, 5, 9, 9), "fork", 0.2),  # below threshold -> dropped
    ]
    dishes = dishes_from_detections(raw, min_conf=0.4)
    assert len(dishes) == 1
    assert dishes[0].label == "plate"
    assert dishes[0].bbox == (0, 0, 20, 20)
    assert dishes[0].confidence == 0.8
    assert dishes[0].id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dish_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dishcounter.dish_detector'`.

- [ ] **Step 3: Create the module**

Create `src/dishcounter/dish_detector.py`:
```python
"""Dish detection behind a Protocol. The Ultralytics YOLO-World implementation
is the only file that imports ultralytics; swapping detectors is a one-file
change. The dish detector is a PRESENCE check only — it never counts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from dishcounter.domain import Dish

Detection = tuple[tuple[int, int, int, int], str, float]  # bbox, label, conf


@runtime_checkable
class DishDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Dish]: ...


class FakeDishDetector:
    """Deterministic detector for tests/headless runs."""

    def __init__(self, script: list[list[Dish]]) -> None:
        if not script:
            script = [[]]
        self._script = script
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Dish]:
        dishes = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return dishes


def dishes_from_detections(detections: list[Detection], min_conf: float) -> list[Dish]:
    """Pure mapping from raw (bbox, label, conf) tuples to Dish objects."""
    return [
        Dish(id=None, bbox=bbox, label=label, confidence=conf)
        for bbox, label, conf in detections
        if conf >= min_conf
    ]


class YoloWorldDishDetector:
    """Real detector using Ultralytics open-vocabulary YOLO-World."""

    def __init__(
        self,
        classes: list[str],
        conf: float,
        model: str = "yolov8s-worldv2.pt",
    ) -> None:  # pragma: no cover - loads model weights
        from ultralytics import YOLOWorld  # noqa: PLC0415

        self._model = YOLOWorld(model)
        self._model.set_classes(classes)
        self._conf = conf

    def detect(self, frame: np.ndarray) -> list[Dish]:  # pragma: no cover - model I/O
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2, y2), label, conf))
        return dishes_from_detections(detections, self._conf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dish_detector.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/dish_detector.py tests/test_dish_detector.py
git commit -m "feat: recover DishDetector protocol, fake, and YOLO-World wrapper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add dish config fields (backward-compatible)

**Files:**
- Modify: `src/dishcounter/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Thresholds` gains `dish_interval: float = 0.5` and `dish_min_hits: int = 1`.
  - `Config` gains `dish_classes: list[str]` (default the 8-item list), `dish_conf: float = 0.4`, `yolo_model: str = "yolov8s-worldv2.pt"`.
  - An existing `config.yaml` lacking all dish fields still loads (defaults fill in).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:
```python
def test_dish_fields_have_defaults():
    from dishcounter.config import Thresholds

    t = Thresholds()
    assert t.dish_interval == 0.5
    assert t.dish_min_hits == 1


def test_config_without_dish_fields_still_loads(tmp_path):
    # An old config (sink zone + a couple thresholds, no dish fields) must load.
    import yaml

    from dishcounter.config import Config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "camera_index": 0,
                "sink_zone": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                "thresholds": {"min_wash": 3.0, "cooldown": 3.0},
            }
        )
    )
    cfg = Config.load(path)
    assert cfg.dish_conf == 0.4
    assert cfg.yolo_model == "yolov8s-worldv2.pt"
    assert "plate" in cfg.dish_classes
    assert cfg.thresholds.dish_min_hits == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py -v -k dish`
Expected: FAIL with `AttributeError: 'Thresholds' object has no attribute 'dish_interval'`.

- [ ] **Step 3: Add the fields**

In `src/dishcounter/config.py`, add `Field` to the pydantic import:
```python
from pydantic import BaseModel, Field
```

Add two fields to `Thresholds` (after the existing ones):
```python
    dish_interval: float = 0.5       # min seconds between dish-detector runs
    dish_min_hits: int = 1           # dish-in-sink confirmations to validate a wash
```

Add three fields to `Config` (after `thresholds`):
```python
    dish_classes: list[str] = Field(
        default_factory=lambda: [
            "plate", "bowl", "cup", "glass", "mug", "fork", "knife", "spoon",
        ]
    )
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (all config tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/config.py tests/test_config.py
git commit -m "feat: add dish-detection config fields with back-compat defaults

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Gate `WashCycleEngine` on dish presence

**Files:**
- Modify: `src/dishcounter/washcycle.py`
- Test: `tests/test_washcycle.py` (replace the file)

**Interfaces:**
- Consumes: `Hand`, `WashEvent`, `Zone` (existing).
- Produces:
  - `WashCycleEngine(sink, min_wash=3.0, cooldown=3.0, dish_min_hits=1)`.
  - `process(hands, now, active_washer, dish_seen: bool = False) -> list[WashEvent]`.
  - Fire requires `dwell >= min_wash` AND `dish_hits >= dish_min_hits` AND active session AND cooldown. `dish_min_hits=0` disables the gate.

- [ ] **Step 1: Replace the test file with the gated tests**

Replace `tests/test_washcycle.py` entirely with:
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


def test_dwell_with_dish_seen_counts_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    ev = eng.process([_out(1)], 3.5, "You")
    assert len(ev) == 1 and ev[0].person == "You" and ev[0].source_id == 1
    assert ev[0].confidence == 1.0


def test_dwell_without_any_dish_seen_does_not_count():
    # The fix: a hand dwell-and-leave with no dish in the sink is NOT a wash.
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")          # dish_seen defaults False
    eng.process([_in(1)], 3.0, "You")
    assert eng.process([_out(1)], 3.5, "You") == []


def test_dish_min_hits_zero_disables_the_gate():
    # Escape hatch: behaves like the old hand-only counter.
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0, dish_min_hits=0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    assert len(eng.process([_out(1)], 3.5, "You")) == 1


def test_dwell_too_short_does_not_count_even_with_dish():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 2.0, "You", dish_seen=True)
    assert eng.process([_out(1)], 2.5, "You") == []


def test_hand_disappearing_after_dish_dwell_counts():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    assert len(eng.process([], 3.5, "You")) == 1


def test_two_hands_leaving_together_collapse_to_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1), _hand(60, 50, 2)], 0.0, "You", dish_seen=True)
    eng.process([_in(1), _hand(60, 50, 2)], 3.0, "You", dish_seen=True)
    assert len(eng.process([], 3.5, "You")) == 1


def test_no_session_counts_nothing():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, None, dish_seen=True)
    eng.process([_in(1)], 3.0, None, dish_seen=True)
    assert eng.process([], 3.5, None) == []


def test_re_entry_needs_its_own_dish_confirmation():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    eng.process([], 3.5, "You")                      # first count (dish_hits reset)
    eng.process([_in(1)], 8.0, "You")                # new visit, NO dish this time
    eng.process([_in(1)], 11.0, "You")
    assert eng.process([], 11.5, "You") == []         # not counted: no dish seen


def test_hand_never_leaving_sink_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 5.0, "You", dish_seen=True)
    assert eng.process([_in(1)], 10.0, "You", dish_seen=True) == []


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
        count += len(eng.process(hands, now, "You", dish_seen=True))
    assert count == 1


def test_idle_out_of_sink_hands_do_not_accumulate_state():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    for i in range(50):
        eng.process([_out(i)], float(i), "You")
    assert eng._visits == {}  # no dead entries left behind
```

- [ ] **Step 2: Run tests to verify the new behavior fails**

Run: `.venv/bin/pytest tests/test_washcycle.py -v`
Expected: FAIL — e.g. `test_dwell_without_any_dish_seen_does_not_count` fails (still counts) and `process()` rejects the `dish_seen` keyword (`TypeError`).

- [ ] **Step 3: Add the gate to `WashCycleEngine`**

Replace the contents of `src/dishcounter/washcycle.py` with:
```python
"""Counting rule: one wash = a hand dwells in the sink for >= min_wash seconds
WITH a dish confirmed present, then leaves it. Credited to the active session
washer; a per-person cooldown collapses two-handed exits."""

from __future__ import annotations

from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Hand, WashEvent


@dataclass
class _Visit:
    enter_time: float | None = None  # start of the current in-sink visit
    last_in_sink: float = 0.0        # last time the hand was seen in the sink
    dish_hits: int = 0               # dish-in-sink confirmations this visit


class WashCycleEngine:
    def __init__(
        self,
        sink: Zone,
        min_wash: float = 3.0,
        cooldown: float = 3.0,
        dish_min_hits: int = 1,
    ) -> None:
        self._sink = sink
        self._min_wash = min_wash
        self._cooldown = cooldown
        self._dish_min_hits = dish_min_hits
        self._visits: dict[int, _Visit] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self,
        hands: list[Hand],
        now: float,
        active_washer: str | None,
        dish_seen: bool = False,
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
                if dish_seen:
                    visit.dish_hits += 1
            else:
                self._close(visit, hand.id, now, active_washer, events)

        for hid in list(self._visits):
            if hid in present:
                continue
            self._close(self._visits[hid], hid, now, active_washer, events)
            del self._visits[hid]
        self._visits = {
            hid: v for hid, v in self._visits.items() if v.enter_time is not None
        }
        return events

    def _close(self, visit: _Visit, hand_id: int, now: float,
               active_washer: str | None, events: list[WashEvent]) -> None:
        if visit.enter_time is None:
            return
        dwell = visit.last_in_sink - visit.enter_time
        dish_hits = visit.dish_hits
        visit.enter_time = None  # close the visit regardless of outcome
        visit.dish_hits = 0
        if dwell < self._min_wash or active_washer not in ("You", "Wife"):
            return
        if dish_hits < self._dish_min_hits:
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_washcycle.py -v`
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/washcycle.py tests/test_washcycle.py
git commit -m "feat: gate wash counting on dish presence in the sink

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wire throttled dish detection into the Engine

**Files:**
- Modify: `src/dishcounter/engine.py`
- Test: `tests/test_engine_e2e.py`

**Interfaces:**
- Consumes: `WashCycleEngine.process(hands, now, active, dish_seen)` (Task 4), `Config.thresholds.dish_interval`/`dish_min_hits` (Task 3), a duck-typed dish detector with `detect(frame) -> list[Dish]`.
- Produces:
  - `Engine(camera, detector, config, store, state, clock=…, jpeg_encoder=…, annotator=…, dish_detector=None)`.
  - When `dish_detector is None`, the gate is off (`WashCycleEngine` built with `dish_min_hits=0`) — behaves like before. When provided, `dish_min_hits` comes from config.
  - The dish detector runs only when a session is active AND a tracked hand is in the sink AND `>= dish_interval` since the last run; the result feeds `dish_seen` into the counter.

- [ ] **Step 1: Add failing tests**

In `tests/test_engine_e2e.py`, add `from dishcounter.dish_detector import FakeDishDetector` to the imports, and change the existing `from dishcounter.domain import Hand` line to:
```python
from dishcounter.dish_detector import FakeDishDetector
from dishcounter.domain import Dish, Hand
```

Change the `_engine` helper to accept a dish detector:
```python
def _engine(cfg, hand_script, frames, times, dish_detector=None):
    return Engine(
        FakeCamera(frames),
        FakeHandDetector(hand_script),
        cfg,
        CountStore(":memory:"),
        SharedState(),
        clock=_fake_clock(times),
        jpeg_encoder=lambda frame: b"jpeg",
        dish_detector=dish_detector,
    )
```

Append these tests:
```python
_DISH_IN_SINK = Dish(id=None, bbox=(40, 40, 60, 60), label="plate", confidence=0.9)


def test_wash_with_dish_in_sink_counts_one(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})                          # gesture, outside sink
    wash = _hand(50, 50, {"index", "middle", "ring"})         # 'other', in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
        dish_detector=FakeDishDetector([[_DISH_IN_SINK]]),  # always sees a plate
    )
    engine.run()
    assert engine._store.totals(8.0)["all_time"] == {"You": 1, "Wife": 0}


def test_same_wash_motion_without_a_dish_counts_zero(sample_config, blank_frame):
    # Core regression: identical hand motion, but the dish detector sees nothing.
    one = _hand(150, 150, {"index"})
    wash = _hand(50, 50, {"index", "middle", "ring"})
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
        dish_detector=FakeDishDetector([[]]),  # never sees a dish
    )
    engine.run()
    assert engine._store.totals(8.0)["all_time"] == {"You": 0, "Wife": 0}


def test_dish_detector_runs_only_in_sink_and_throttled(sample_config, blank_frame):
    class _Spy:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return [_DISH_IN_SINK]

    spy = _Spy()
    one = _hand(150, 150, {"index"})                   # outside sink (gesture)
    wash = _hand(50, 50, {"index", "middle", "ring"})  # in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], [wash]],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.2, 1.4, 2.0],
        dish_detector=spy,
    )
    for now in [0.0, 1.0, 1.2, 1.4, 2.0]:
        engine.process_frame(blank_frame, now)
    # Not called on the two gesture frames (no hand in sink). In-sink frames at
    # 1.2 (first), 1.4 (throttled out: <0.5s later), 2.0 (>=0.5s later) -> 2 runs.
    assert spy.calls == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_engine_e2e.py -v -k "dish or throttled"`
Expected: FAIL — `Engine.__init__()` rejects `dish_detector` (`TypeError`).

- [ ] **Step 3: Add dish detection to the Engine**

In `src/dishcounter/engine.py`, change the `Engine.__init__` signature to add the parameter (keep all existing params; append `dish_detector`):
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
        dish_detector=None,
    ) -> None:
```

In the body of `__init__`, store the detector and throttle state, and gate the counter's `dish_min_hits` (replace the `self._counter = …` line):
```python
        self._dish_detector = dish_detector
        self._dish_interval = t.dish_interval
        self._last_dish_run: float | None = None
        dish_min_hits = t.dish_min_hits if dish_detector is not None else 0
        self._counter = WashCycleEngine(
            config.sink_zone, min_wash=t.min_wash, cooldown=t.cooldown,
            dish_min_hits=dish_min_hits,
        )
```
(`self._dish_detector`/`self._dish_interval`/`self._last_dish_run` must be assigned **before** any use; place them with the other `self._…` assignments. `t = config.thresholds` already exists above.)

Add a method (after `_resolve_gesture`):
```python
    def _maybe_detect_dish(self, frame, hands, active, now) -> bool:
        # Run YOLO only when a wash could be happening, throttled to dish_interval.
        if self._dish_detector is None or active not in ("You", "Wife"):
            return False
        sink = self._config.sink_zone
        if not any(h.id is not None and sink.contains(h.centroid) for h in hands):
            return False
        if self._last_dish_run is not None and (now - self._last_dish_run) < self._dish_interval:
            return False
        self._last_dish_run = now
        try:
            dishes = self._dish_detector.detect(frame)
        except Exception:
            return False
        return any(sink.contains(d.centroid) for d in dishes)
```

In `process_frame`, compute `dish_seen` and pass it to the counter (replace the `events = self._counter.process(...)` line and add the line above it):
```python
        dish_seen = self._maybe_detect_dish(frame, hands, active, now)
        events = self._counter.process(hands, now, active, dish_seen)
```
(Leave the `self._annotate(frame, self._config, hands, active, gesture)` call unchanged in this task — Task 6 adds the indicator.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine_e2e.py -v`
Expected: PASS — the new dish tests pass, and the pre-existing e2e tests (which pass no `dish_detector`, so the gate is off) still pass.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/dishcounter/engine.py tests/test_engine_e2e.py
git commit -m "feat: run throttled dish detection in the engine to gate washes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Show a dish indicator on the overlay

**Files:**
- Modify: `src/dishcounter/engine.py` (the `annotate` function + its call in `process_frame`)
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `annotate(frame, config, hands, active_washer, gesture, dish_seen: bool = False)` — draws a small "dish ✓" marker when `dish_seen` is True. `process_frame` passes its computed `dish_seen`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_annotate.py`:
```python
def test_annotate_with_dish_seen_draws_marker():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]
    base = annotate(frame, _cfg(), hands, "You", "one", False)
    marked = annotate(frame, _cfg(), hands, "You", "one", True)
    assert marked.shape == frame.shape
    # The dish marker adds pixels the un-marked frame does not have.
    assert int(marked.sum()) > int(base.sum())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_annotate.py::test_annotate_with_dish_seen_draws_marker -v`
Expected: FAIL with `TypeError` (annotate takes 5 positional args, 6 given).

- [ ] **Step 3: Add the `dish_seen` marker**

In `src/dishcounter/engine.py`, change the `annotate` signature to add the parameter:
```python
def annotate(frame: np.ndarray, config: Config, hands, active_washer, gesture,
             dish_seen: bool = False) -> np.ndarray:
```

Just before `return out` in `annotate`, add:
```python
    if dish_seen:
        cv2.putText(out, "dish detected", (8, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
```

In `process_frame`, pass `dish_seen` to the annotate call (replace the existing call):
```python
        annotated = self._annotate(
            frame, self._config, hands, active, gesture, dish_seen
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_annotate.py -v`
Expected: PASS (all annotate tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/engine.py tests/test_annotate.py
git commit -m "feat: show a 'dish detected' marker on the live overlay

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Wire YOLO-World into the CLI and add the dependency

**Files:**
- Modify: `src/dishcounter/cli.py` (the `_serve` function)
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `YoloWorldDishDetector` (Task 2), `Config.dish_classes/dish_conf/yolo_model` (Task 3), `Engine(..., dish_detector=…)` (Task 5).
- Produces: a `run` command that builds the real dish detector and passes it to the Engine.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` list (after `"pyyaml>=6.0",`):
```python
    "ultralytics>=8.1",  # YOLO-World open-vocabulary dish detection (presence gate)
```

In `requirements.txt`, add a final line:
```
ultralytics>=8.1
```

- [ ] **Step 2: Build the detector in `_serve`**

In `src/dishcounter/cli.py`, inside `_serve`, add the import alongside the other lazy imports:
```python
    from dishcounter.dish_detector import YoloWorldDishDetector  # noqa: PLC0415
```

Replace the `engine = Engine(...)` construction with one that passes the dish detector:
```python
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        config,
        CountStore(db),
        state,
        dish_detector=YoloWorldDishDetector(
            config.dish_classes, config.dish_conf, config.yolo_model
        ),
    )
```

- [ ] **Step 3: Verify nothing else broke (no new unit test; `_serve` is `# pragma: no cover`)**

Run:
```bash
.venv/bin/python -c "import dishcounter.cli"
.venv/bin/dishcounter --help
.venv/bin/pytest -q
```
Expected: import succeeds; help prints `calibrate` and `run`; the full suite passes (no test imports `ultralytics`).

- [ ] **Step 4: Commit**

```bash
git add src/dishcounter/cli.py pyproject.toml requirements.txt
git commit -m "feat: build YOLO-World dish detector in the run command

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update the README and final sign-off

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: docs explaining the dish-presence gate; a green full suite and clean lint.

- [ ] **Step 1: Document the gate in the README**

In `README.md`, in the bullet list under the title, replace the first bullet (the one beginning "**A wash is counted**") with:
```markdown
- **A wash is counted** when your hand dwells in the **sink zone** for a few seconds,
  **a dish is seen in the sink during that dwell**, and then your hand **leaves**.
  The dish check (a webcam object detector) is what stops hand-rinsing or
  sponge-wringing from being counted — no dish in the sink, no count.
```

In the **Run** section, after the line about MediaPipe downloading its model, add:
```markdown
The dish detector (YOLO-World) also downloads its weights on first run
(~340MB, including the CLIP text encoder). It runs only while a session is active
and a hand is in the sink, so it stays idle when nobody is washing.
```

In the **Tuning** `thresholds` table, add two rows:
```markdown
| `dish_interval` | `0.5` | Minimum seconds between dish-detector runs during a wash | Lower for more frequent dish checks (more CPU); raise to save CPU |
| `dish_min_hits` | `1` | Dish-in-sink confirmations needed to count a wash; `0` disables the dish gate | Raise if you still see false counts; set `0` to count on hand activity alone |
```

- [ ] **Step 2: Final full-suite and lint sign-off**

Run:
```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```
Expected: all tests pass, pristine (no warnings); ruff reports `All checks passed!`.

If either is not clean, STOP and fix before committing — this is a sign-off gate.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: explain the dish-presence gate and its tuning

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

- `Dish` type recovered → Task 1. ✓
- `DishDetector` Protocol / `FakeDishDetector` / `dishes_from_detections` / `YoloWorldDishDetector` recovered → Task 2. ✓
- Config: `dish_classes`/`dish_conf`/`yolo_model` + `dish_interval`/`dish_min_hits`, all defaulted, back-compat load → Task 3. ✓
- Gate rule in `WashCycleEngine` (`dish_hits`, `dish_min_hits`, `dish_seen` arg; fire requires dwell + dish + session + cooldown; `dish_min_hits=0` disables) → Task 4. ✓
- Compute model: dish detector runs only when session active AND hand in sink AND throttled by `dish_interval`; `dish_seen` = dish centroid in sink; detector exception → False; gate off when no detector → Task 5. ✓
- Annotate dish indicator → Task 6. ✓
- cli builds `YoloWorldDishDetector` from config; `ultralytics` dependency → Task 7. ✓
- README gate explanation + weights note + tuning rows → Task 8. ✓
- Bias to undercount, presence-not-tracking, no ultralytics in tests, identity from session, `source_id` → enforced across Tasks 4/5 and Global Constraints. ✓
- Core regression (same hand motion, no dish → zero) → Task 4 (`test_dwell_without_any_dish_seen_does_not_count`) and Task 5 (`test_same_wash_motion_without_a_dish_counts_zero`). ✓

**Placeholder scan:** No TBD/TODO/"similar to"/"add error handling". Every code step shows complete code.

**Type consistency:** `process(hands, now, active_washer, dish_seen=False)` is used identically in Task 4 (definition) and Task 5 (engine call). `WashCycleEngine(..., dish_min_hits=…)` matches between Task 4 and Task 5. `Dish(id, bbox, label, confidence)` + `.centroid` consistent across Tasks 1, 2, 5. `Engine(..., dish_detector=None)` matches between Task 5 (def), Task 5 tests, and Task 7 (cli). `annotate(..., dish_seen=False)` matches between Task 6 (def) and the `process_frame` call. Config field names (`dish_interval`, `dish_min_hits`, `dish_classes`, `dish_conf`, `yolo_model`) consistent across Tasks 3, 5, 7.
