# Dish-Detection Counting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count each washed dish when the detected dish object leaves the frame after being washed at the sink, attributing it to You/Wife via the nearest hand's skin tone.

**Architecture:** Two detectors run per frame — YOLO-World detects dishes (the count), MediaPipe detects hands (the identity). Each is tracked by a shared IoU tracker. A `FusionEngine` locks each sink-visited dish's identity to the nearest washing hand and fires one `WashEvent` when that dish disappears for longer than a grace period. Events flow unchanged into the event-sourced SQLite `CountStore`.

**Tech Stack:** Python 3.12, Ultralytics YOLO-World, MediaPipe, OpenCV, Pydantic, pytest.

## Global Constraints

- Python `>=3.12,<3.13`; ruff `line-length = 100`.
- Detection runs behind dependency-injected Protocols; **no real model loads in tests** — tests use Fake detectors only (mirror `FakeCamera`/`FakeHandDetector`).
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Run tests with `PYTHONPATH=src python -m pytest` (package is not pip-installed in the dev shell).
- Domain types are the only types crossing component boundaries; downstream code never sees MediaPipe/Ultralytics types.
- Identity stays biased to `uncertain`; `uncertain` events are recorded but not counted by the store.
- New dependency allowed: `ultralytics`. No other new runtime deps.

---

### Task 1: `Dish` domain type + rename `WashEvent.hand_id` → `source_id`

**Files:**
- Modify: `src/dishcounter/domain.py`
- Modify: `src/dishcounter/zones.py:60` (only producer of `WashEvent`; updates the keyword) — note this file is deleted in Task 6, but must stay green until then.
- Test: `tests/test_domain.py`

**Interfaces:**
- Produces: `Dish(id: int | None, bbox: tuple[int,int,int,int], label: str, confidence: float)` with `.centroid -> tuple[float,float]`; `WashEvent(person, timestamp, confidence, source_id)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_domain.py`:

```python
from dishcounter.domain import Dish, WashEvent


def test_dish_centroid_is_bbox_center():
    dish = Dish(id=1, bbox=(10, 20, 30, 40), label="plate", confidence=0.9)
    assert dish.centroid == (20.0, 30.0)


def test_wash_event_has_source_id():
    ev = WashEvent(person="You", timestamp=1.0, confidence=0.5, source_id=7)
    assert ev.source_id == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_domain.py -v`
Expected: FAIL — `ImportError: cannot import name 'Dish'` (and `source_id` unknown).

- [ ] **Step 3: Write minimal implementation**

In `src/dishcounter/domain.py`, rename the field on `WashEvent`:

```python
@dataclass
class WashEvent:
    """A single counted (or uncertain) dish wash."""

    person: str  # "You" | "Wife" | "uncertain"
    timestamp: float
    confidence: float
    source_id: int  # tracker id of the dish (or hand) that produced the event
```

Add the `Dish` type below `Hand`:

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

In `src/dishcounter/zones.py`, change the one producer at line ~60 from `hand_id=hand.id` to `source_id=hand.id` so the existing suite still imports/runs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_domain.py tests/test_zones.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/domain.py src/dishcounter/zones.py tests/test_domain.py
git commit -m "feat: add Dish domain type, rename WashEvent.hand_id to source_id"
```

---

### Task 2: Generalize the IoU tracker to any boxed object

**Files:**
- Modify: `src/dishcounter/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: any object with a `.bbox` tuple and assignable `.id`.
- Produces: `IouTracker(iou_threshold=0.3).update(items) -> items` (sets `.id` on each, stable across frames by IoU). `HandTracker` remains as an alias for back-compat.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tracker.py`:

```python
from dishcounter.domain import Dish
from dishcounter.tracker import IouTracker


def test_tracker_assigns_stable_ids_to_dishes():
    tracker = IouTracker(iou_threshold=0.3)
    d1 = Dish(id=None, bbox=(0, 0, 40, 40), label="plate", confidence=0.9)
    [tracked] = tracker.update([d1])
    first_id = tracked.id
    # Next frame: heavily-overlapping box keeps the same id.
    d2 = Dish(id=None, bbox=(5, 5, 45, 45), label="plate", confidence=0.9)
    [tracked2] = tracker.update([d2])
    assert tracked2.id == first_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_tracker.py::test_tracker_assigns_stable_ids_to_dishes -v`
Expected: FAIL — `ImportError: cannot import name 'IouTracker'`.

- [ ] **Step 3: Write minimal implementation**

In `src/dishcounter/tracker.py`, rename `class HandTracker` to `class IouTracker`, rename the local variables `hands`/`hand` to `items`/`item` in `update`, and update the module docstring to say "any boxed object". At the end of the file add the alias:

```python
HandTracker = IouTracker  # back-compat: hands are just boxed objects
```

The body is otherwise unchanged (it already only uses `.bbox` and assigns `.id`):

```python
class IouTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._iou_threshold = iou_threshold
        self._next_id = 0
        self._tracks: dict[int, Box] = {}  # id -> last bbox

    def update(self, items: list) -> list:
        unmatched_tracks = dict(self._tracks)
        new_tracks: dict[int, Box] = {}

        candidates = [
            (iou(it.bbox, box), idx, tid)
            for idx, it in enumerate(items)
            for tid, box in unmatched_tracks.items()
        ]
        candidates.sort(reverse=True)

        assigned: dict[int, int] = {}
        for score, idx, tid in candidates:
            if score < self._iou_threshold:
                break
            if idx in assigned or tid not in unmatched_tracks:
                continue
            assigned[idx] = tid
            del unmatched_tracks[tid]

        for idx, item in enumerate(items):
            if idx in assigned:
                item.id = assigned[idx]
            else:
                item.id = self._next_id
                self._next_id += 1
            new_tracks[item.id] = item.bbox

        self._tracks = new_tracks
        return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_tracker.py -v`
Expected: PASS (existing `HandTracker` tests pass via the alias, new dish test passes).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/tracker.py tests/test_tracker.py
git commit -m "refactor: generalize HandTracker into IouTracker for any boxed object"
```

---

### Task 3: `DishDetector` Protocol, fake, and YOLO-World wrapper

**Files:**
- Create: `src/dishcounter/dish_detector.py`
- Test: `tests/test_dish_detector.py`

**Interfaces:**
- Produces:
  - `DishDetector` Protocol: `detect(frame) -> list[Dish]`.
  - `FakeDishDetector(script: list[list[Dish]])` — deterministic, mirrors `FakeHandDetector`.
  - `dishes_from_detections(detections: list[tuple[tuple[int,int,int,int], str, float]], min_conf: float) -> list[Dish]` — pure parser.
  - `YoloWorldDishDetector(classes: list[str], conf: float, model: str = "yolov8s-worldv2.pt")` — real wrapper (model I/O is `# pragma: no cover`).

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

Run: `PYTHONPATH=src python -m pytest tests/test_dish_detector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dishcounter.dish_detector'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/dishcounter/dish_detector.py`:

```python
"""Dish detection behind a Protocol. The Ultralytics YOLO-World implementation
is the only file that imports ultralytics; swapping detectors is a one-file
change. Counting is dish-driven; identity comes from hands elsewhere."""

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_dish_detector.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/dish_detector.py tests/test_dish_detector.py
git commit -m "feat: add DishDetector protocol, fake, and YOLO-World wrapper"
```

---

### Task 4: Config — add dish/exit settings (additive, keep drying_zone for now)

**Files:**
- Modify: `src/dishcounter/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Thresholds(identity_distance=25.0, exit_grace=1.5, cooldown=3.0, iou_match=0.3)`; `Config` gains `dish_classes: list[str]`, `dish_conf: float = 0.4`, `yolo_model: str = "yolov8s-worldv2.pt"`. (`presence_window` removed; `drying_zone` still present, removed in Task 6.)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_config_has_dish_settings_and_exit_grace():
    from dishcounter.config import Config, SkinProfile, Zone

    cfg = Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        drying_zone=Zone(x1=100, y1=0, x2=200, y2=100),
        you_profile=SkinProfile(cr=165.0, cb=110.0),
        wife_profile=SkinProfile(cr=120.0, cb=150.0),
    )
    assert "plate" in cfg.dish_classes
    assert cfg.dish_conf == 0.4
    assert cfg.yolo_model == "yolov8s-worldv2.pt"
    assert cfg.thresholds.exit_grace == 1.5
    assert not hasattr(cfg.thresholds, "presence_window")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError`/validation: `dish_classes` and `exit_grace` not defined.

- [ ] **Step 3: Write minimal implementation**

In `src/dishcounter/config.py`, replace `Thresholds` and add fields to `Config`:

```python
DEFAULT_DISH_CLASSES = ["plate", "bowl", "cup", "glass", "mug", "fork", "knife", "spoon"]


class Thresholds(BaseModel):
    identity_distance: float = 25.0  # max YCrCb distance to accept an identity
    exit_grace: float = 1.5          # seconds a dish must be gone before counting
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id


class Config(BaseModel):
    camera_index: int
    sink_zone: Zone
    drying_zone: Zone  # removed in Task 6
    you_profile: SkinProfile
    wife_profile: SkinProfile
    dish_classes: list[str] = DEFAULT_DISH_CLASSES
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"
    thresholds: Thresholds = Thresholds()
```

(Keep the existing `load`/`save` methods unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/config.py tests/test_config.py
git commit -m "feat: add dish-detection config fields and exit_grace threshold"
```

---

### Task 5: `FusionEngine` — dish-exit counting with nearest-hand identity

**Files:**
- Create: `src/dishcounter/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `Zone`, `IdentityClassifier` (`.classify(hand) -> (str, float)`), `Hand`, `Dish`, `WashEvent`.
- Produces: `FusionEngine(sink, identity, exit_grace=1.5, cooldown=3.0).process(hands, dishes, now) -> list[WashEvent]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fusion.py`:

```python
import numpy as np

from dishcounter.config import SkinProfile, Zone
from dishcounter.domain import Dish, Hand
from dishcounter.fusion import FusionEngine
from dishcounter.identity import IdentityClassifier

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))   # -> You
BLUE = np.tile(np.array([220, 90, 90], dtype=np.uint8), (16, 1))  # -> Wife
SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _identity():
    return IdentityClassifier(
        SkinProfile(cr=165.0, cb=110.0), SkinProfile(cr=120.0, cb=150.0),
        max_distance=60.0,
    )


def _hand(cx, region, hid):
    return Hand(id=hid, bbox=(cx - 10, 40, cx + 10, 60), region_pixels=region)


def _dish(cx, did, in_sink=True):
    cy = 50 if in_sink else 200
    return Dish(id=did, bbox=(cx - 10, cy - 10, cx + 10, cy + 10), label="plate",
                confidence=0.9)


def test_sink_dish_that_leaves_fires_one_event_for_nearest_hand():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    assert eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0) == []
    events = eng.process([], [], now=2.0)  # dish gone for 2.0s > grace
    assert len(events) == 1
    assert events[0].person == "You"
    assert events[0].source_id == 1


def test_flicker_under_grace_does_not_fire():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0)
    eng.process([], [], now=0.5)                      # brief dropout
    events = eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=1.0)  # back
    events += eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=1.2)
    assert events == []


def test_dish_never_in_sink_does_not_count():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1, in_sink=False)], now=0.0)
    assert eng.process([], [], now=2.0) == []


def test_two_dishes_same_person_within_cooldown_collapse_to_one():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(40, 1), _dish(60, 2)], now=0.0)
    events = eng.process([], [], now=2.0)  # both gone together
    assert len(events) == 1
    assert events[0].person == "You"


def test_dish_washed_with_no_confident_hand_is_uncertain():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([], [_dish(50, 1)], now=0.0)  # no hands near
    events = eng.process([], [], now=2.0)
    assert len(events) == 1
    assert events[0].person == "uncertain"


def test_reappearing_dish_with_new_id_counts_again():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0)
    first = eng.process([], [], now=2.0)               # fires You
    eng.process([_hand(50, RED, 2)], [_dish(50, 2)], now=6.0)  # new dish id
    second = eng.process([], [], now=8.0)              # past cooldown -> fires
    assert len(first) == 1 and len(second) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dishcounter.fusion'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/dishcounter/fusion.py`:

```python
"""The counting rule. Dishes drive the count: a dish that was washed at the
sink and then disappears from view is one washed dish. Hands only supply the
You/Wife identity (nearest hand at the sink)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Dish, Hand, WashEvent
from dishcounter.identity import IdentityClassifier


@dataclass
class _DishState:
    last_sink_time: float | None = None
    last_seen: float = 0.0
    locked_person: str = "uncertain"
    locked_confidence: float = 0.0


class FusionEngine:
    def __init__(
        self,
        sink: Zone,
        identity: IdentityClassifier,
        exit_grace: float = 1.5,
        cooldown: float = 3.0,
    ) -> None:
        self._sink = sink
        self._identity = identity
        self._exit_grace = exit_grace
        self._cooldown = cooldown
        self._states: dict[int, _DishState] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self, hands: list[Hand], dishes: list[Dish], now: float
    ) -> list[WashEvent]:
        present: set[int] = set()
        for dish in dishes:
            if dish.id is None:
                continue
            present.add(dish.id)
            state = self._states.setdefault(dish.id, _DishState())
            state.last_seen = now
            if self._sink.contains(dish.centroid):
                state.last_sink_time = now
                if state.locked_person == "uncertain":
                    self._lock_identity(state, dish, hands)

        events: list[WashEvent] = []
        for did in list(self._states):
            if did in present:
                continue
            state = self._states[did]
            if (now - state.last_seen) <= self._exit_grace:
                continue
            if state.last_sink_time is not None and self._can_fire(
                state.locked_person, now
            ):
                events.append(
                    WashEvent(
                        person=state.locked_person,
                        timestamp=now,
                        confidence=state.locked_confidence,
                        source_id=did,
                    )
                )
                self._last_person_fire[state.locked_person] = now
            del self._states[did]
        return events

    def _lock_identity(self, state: _DishState, dish: Dish, hands: list[Hand]) -> None:
        hand = self._nearest_hand(dish, hands)
        if hand is None:
            return
        label, conf = self._identity.classify(hand)
        if label != "uncertain":
            state.locked_person = label
            state.locked_confidence = conf

    @staticmethod
    def _nearest_hand(dish: Dish, hands: list[Hand]) -> Hand | None:
        if not hands:
            return None
        return min(hands, key=lambda h: math.dist(h.centroid, dish.centroid))

    def _can_fire(self, person: str, now: float) -> bool:
        if person not in ("You", "Wife"):
            return True  # uncertain: recorded (not counted by the store), no cooldown
        last = self._last_person_fire.get(person)
        return last is None or (now - last) >= self._cooldown
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_fusion.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/fusion.py tests/test_fusion.py
git commit -m "feat: add FusionEngine for dish-exit counting with nearest-hand identity"
```

---

### Task 6: Cutover — wire two detectors into the Engine, annotate dishes, retire zones

**Files:**
- Modify: `src/dishcounter/engine.py`
- Modify: `src/dishcounter/cli.py:60-67` (construct the dish detector)
- Modify: `tests/conftest.py` (drop `drying_zone` from fixture)
- Modify: `tests/test_engine_e2e.py` (rewrite for dish-exit counting)
- Delete: `src/dishcounter/zones.py`, `tests/test_zones.py`
- Modify: `src/dishcounter/config.py` (remove `drying_zone` field)
- Modify: `tests/test_config.py` (drop `drying_zone` from the constructor call added in Task 4)

**Interfaces:**
- Consumes: `IouTracker`, `FusionEngine`, `DishDetector`/`FakeDishDetector`, `YoloWorldDishDetector`.
- Produces: `Engine(camera, detector, dish_detector, config, store, state, clock=..., jpeg_encoder=..., annotator=...)`; `annotate(frame, config, hands, dishes) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Replace the body of `tests/test_engine_e2e.py` with:

```python
import numpy as np

from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.dish_detector import FakeDishDetector
from dishcounter.domain import Dish, Hand
from dishcounter.engine import Engine
from dishcounter.state import SharedState
from dishcounter.store import CountStore

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))   # -> You
BLUE = np.tile(np.array([220, 90, 90], dtype=np.uint8), (16, 1))  # -> Wife


def _hand(cx, region):
    return Hand(id=None, bbox=(cx - 20, 40, cx + 20, 60), region_pixels=region)


def _dish(cx):
    return Dish(id=None, bbox=(cx - 20, 40, cx + 20, 60), label="plate", confidence=0.9)


def _fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def _engine(sample_config, hand_script, dish_script, frames, times):
    return Engine(
        FakeCamera(frames),
        FakeHandDetector(hand_script),
        FakeDishDetector(dish_script),
        sample_config,
        CountStore(":memory:"),
        SharedState(),
        clock=_fake_clock(times),
        jpeg_encoder=lambda frame: b"jpeg",
    )


def test_full_pipeline_counts_one_wash_for_you(sample_config, blank_frame):
    # Frame 0: hand + dish in the sink (cx=50, sink zone 0..100) -> lock You.
    # Frame 1: both gone; 2.0s > exit_grace (1.5) -> fire one You wash.
    engine = _engine(
        sample_config,
        hand_script=[[_hand(50, RED)], []],
        dish_script=[[_dish(50)], []],
        frames=[blank_frame, blank_frame],
        times=[100.0, 102.0],
    )
    engine.run()  # FakeCamera exhausts after 2 frames

    assert engine_store_totals(engine) == {"You": 1, "Wife": 0}


def engine_store_totals(engine):
    return engine._store.totals(10_000_000_000.0)["all_time"]


def test_process_frame_publishes_counts(sample_config, blank_frame):
    engine = _engine(
        sample_config,
        hand_script=[[_hand(50, BLUE)], []],
        dish_script=[[_dish(50)], []],
        frames=[],
        times=[],
    )
    engine.process_frame(blank_frame, now=0.0)        # dish in sink -> lock Wife
    events = engine.process_frame(blank_frame, now=2.0)  # dish gone -> fire
    assert len(events) == 1
    assert events[0].person == "Wife"
```

(`sample_config`/`blank_frame` come from `conftest.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_engine_e2e.py -v`
Expected: FAIL — `Engine.__init__` does not accept `dish_detector`; `annotate` arity.

- [ ] **Step 3: Write minimal implementation**

In `src/dishcounter/engine.py`:

1. Update imports — replace the zones import:

```python
from dishcounter.dish_detector import DishDetector
from dishcounter.fusion import FusionEngine
from dishcounter.tracker import IouTracker
```

2. Update `annotate` to draw the sink zone and dish boxes:

```python
def annotate(frame: np.ndarray, config: Config, hands, dishes) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    z = config.sink_zone
    cv2.rectangle(out, (z.x1, z.y1), (z.x2, z.y2), (255, 0, 0), 2)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    for dish in dishes:
        x1, y1, x2, y2 = dish.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, dish.label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
```

3. Update `Engine.__init__` signature and body — add `dish_detector`, build two trackers and the `FusionEngine`, drop the old tracker/zones wiring:

```python
    def __init__(
        self,
        camera,
        detector,
        dish_detector: DishDetector,
        config: Config,
        store: CountStore,
        state: SharedState,
        clock: Callable[[], float] = time.time,
        jpeg_encoder: Callable[[np.ndarray], bytes] = encode_jpeg,
        annotator: Callable = annotate,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._dish_detector = dish_detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._running = False

        t = config.thresholds
        self._hand_tracker = IouTracker(iou_threshold=t.iou_match)
        self._dish_tracker = IouTracker(iou_threshold=t.iou_match)
        self._identity = IdentityClassifier(
            config.you_profile, config.wife_profile, max_distance=t.identity_distance
        )
        self._fusion = FusionEngine(
            config.sink_zone,
            self._identity,
            exit_grace=t.exit_grace,
            cooldown=t.cooldown,
        )
```

4. Update `process_frame`:

```python
    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame))
        dishes = self._dish_tracker.update(self._dish_detector.detect(frame))
        events = self._fusion.process(hands, dishes, now)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands, dishes)
        self._state.publish(
            self._encode(annotated), self._store.totals(now), camera_online=True
        )
        return events
```

5. In `src/dishcounter/cli.py`, import and construct the dish detector:

```python
    from dishcounter.dish_detector import YoloWorldDishDetector  # noqa: PLC0415
    ...
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        YoloWorldDishDetector(config.dish_classes, config.dish_conf, config.yolo_model),
        config,
        CountStore(db),
        state,
    )
```

6. In `tests/conftest.py`, remove the `drying_zone=...` line from `sample_config`.

7. In `src/dishcounter/config.py`, delete the `drying_zone: Zone` field. In `tests/test_config.py`, remove the `drying_zone=...` argument from the constructor call added in Task 4.

8. Delete the retired files:

```bash
git rm src/dishcounter/zones.py tests/test_zones.py
```

- [ ] **Step 4: Run the full suite to verify it passes**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS (all tests green; no references to `drying_zone`/`zones`/`presence_window` remain).

Sanity grep — expect no output:

Run: `grep -rn "drying\|presence_window\|ZoneEventEngine\|from dishcounter.zones" src/ tests/`
Expected: (no matches)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: cut over Engine to dish-detection fusion counting, retire zones"
```

---

### Task 7: Calibration — sink zone only

**Files:**
- Modify: `src/dishcounter/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `Config` without `drying_zone`.
- Produces: a saved `Config` built from one calibrated `sink` zone.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibrate.py` (and check the file for any existing test that constructs a `Config`/`Zone` named `drying` — update those to drop it):

```python
def test_calibration_module_does_not_reference_drying_zone():
    import inspect

    from dishcounter import calibrate

    source = inspect.getsource(calibrate)
    assert "drying" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_calibrate.py -v`
Expected: FAIL — `calibrate.py` still contains "drying".

- [ ] **Step 3: Write minimal implementation**

In `src/dishcounter/calibrate.py`: update the docstring to say "sink rectangle" (singular), change the zone loop to only `("sink",)`, and drop `drying_zone` from the `Config(...)` call:

```python
    zones: dict[str, Zone] = {}
    for label in ("sink",):
        frame = None
        while frame is None:
            frame = cam.read()
        roi = cv2.selectROI(f"drag the {label} zone", frame, showCrosshair=True)
        x, y, bw, bh = (int(v) for v in roi)
        zones[label] = zone_from_drag((x, y), (x + bw, y + bh))
    cv2.destroyAllWindows()
    cam.release()

    config = Config(
        camera_index=camera_index,
        sink_zone=zones["sink"],
        you_profile=profiles["You"],
        wife_profile=profiles["Wife"],
        thresholds=Thresholds(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_calibrate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/calibrate.py tests/test_calibrate.py
git commit -m "feat: calibrate only the sink zone (drying zone removed)"
```

---

### Task 8: Dependency + docs

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `README.md` (run section: note first-run weight download)

**Interfaces:** none (packaging/docs).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to `dependencies` (after `pydantic>=2.6`):

```toml
    "ultralytics>=8.2",
```

In `requirements.txt`, add a line:

```
ultralytics>=8.2
```

- [ ] **Step 2: Document first-run behavior**

In `README.md`, under the run instructions, add a short note:

> On first `dishcounter run`, the YOLO-World weights (`yolov8s-worldv2.pt`) download automatically (~25 MB). Dish classes are configurable in `config.yaml` (`dish_classes`).

- [ ] **Step 3: Verify the full suite still passes**

Run: `PYTHONPATH=src python -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml requirements.txt README.md
git commit -m "chore: add ultralytics dependency, document YOLO-World weight download"
```

---

## Self-Review

**Spec coverage:**
- YOLO-World dish detection (no training, detects plate) → Task 3 + Task 8.
- MediaPipe hands supply identity → reused; nearest-hand lock in Task 5.
- Two trackers (hands + dishes) → Task 2 + Task 6.
- FusionEngine dish-exit trigger, sink gate, exit_grace, per-person cooldown, uncertain-recorded-not-counted, bounded state → Task 5.
- Config: drop `drying_zone`/`presence_window`, add `exit_grace`, `dish_classes`, `dish_conf`, `yolo_model` → Task 4 + Task 6.
- Engine wiring + annotate dishes, drop drying box → Task 6.
- Calibrate sink only → Task 7.
- Dependency + README → Task 8.
- All detection behind DI fakes; no model in tests → Tasks 3, 5, 6.

**Placeholder scan:** none — every code/test step contains complete code; commands have expected output.

**Type consistency:** `Dish`, `WashEvent.source_id`, `IouTracker.update`, `dishes_from_detections`, `FusionEngine(sink, identity, exit_grace, cooldown).process(hands, dishes, now)`, and `Engine(camera, detector, dish_detector, config, store, state, ...)` / `annotate(frame, config, hands, dishes)` are used identically across the tasks that define and consume them.
