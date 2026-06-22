# Gesture-Controlled Session Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace skin-tone identity with hand gestures — show 1 finger (You) / 2 fingers (Wife) to start a counting session, a fist to end it; dishes washed during an active session count for that person.

**Architecture:** A pure gesture recognizer reads finger state from existing MediaPipe hand landmarks. A debounced `SessionController` turns held gestures into an `active_washer`. The `FusionEngine` attributes each sink-visited dish to the active washer (or `uncertain` if no session). The skin-tone/ring subsystem is removed.

**Tech Stack:** Python 3.12, MediaPipe (hand landmarks, already integrated), Ultralytics YOLO-World (dishes, unchanged), pytest.

## Global Constraints

- Python `>=3.12,<3.13`; ruff `line-length = 100`.
- All detection behind dependency-injected fakes; **no real model loads in tests**.
- Run tests with `.venv/bin/python -m pytest` (the venv has all deps; the package is installed editable).
- TDD: failing test first, watch it fail, minimal implementation, watch it pass, commit.
- Domain types are the only types crossing component boundaries.
- Identity stays biased to safety: a dish with no active session exits `uncertain` (recorded, `counted=0`), never guessed.
- Gestures: `one` finger = You, `two` = Wife, `fist` = end session. A gesture must be held `gesture_hold` seconds (default 1.0) to act.

---

### Task 1: `gesture.py` — recognize a gesture from hand landmarks

**Files:**
- Create: `src/dishcounter/gesture.py`
- Test: `tests/test_gesture.py`

**Interfaces:**
- Consumes: `Hand.landmarks` — a list of 21 `(x, y)` normalized tuples (empty if unknown).
- Produces: `recognize_gesture(hand) -> str` returning `"one"`, `"two"`, `"fist"`, or `"other"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gesture.py`:

```python
from dishcounter.domain import Hand
from dishcounter.gesture import recognize_gesture

# Finger -> (tip_idx, pip_idx, x). Thumb is ignored.
_FINGERS = {"index": (8, 6, 0.40), "middle": (12, 10, 0.48),
            "ring": (16, 14, 0.56), "pinky": (20, 18, 0.64)}


def _landmarks(extended: set[str], pointing: str = "up") -> list[tuple[float, float]]:
    if pointing == "up":
        wrist, ext_y, curl_y, pip_y = (0.5, 0.9), 0.25, 0.55, 0.5
    else:
        wrist, ext_y, curl_y, pip_y = (0.5, 0.1), 0.75, 0.45, 0.5
    pts = [wrist] * 21
    for name, (tip, pip, x) in _FINGERS.items():
        pts[pip] = (x, pip_y)
        pts[tip] = (x, ext_y if name in extended else curl_y)
    return pts


def _hand(extended: set[str], pointing: str = "up") -> Hand:
    return Hand(id=1, bbox=(0, 0, 10, 10), landmarks=_landmarks(extended, pointing))


def test_one_finger_is_one():
    assert recognize_gesture(_hand({"index"})) == "one"


def test_two_fingers_is_two():
    assert recognize_gesture(_hand({"index", "middle"})) == "two"


def test_no_fingers_is_fist():
    assert recognize_gesture(_hand(set())) == "fist"


def test_three_fingers_is_other():
    assert recognize_gesture(_hand({"index", "middle", "ring"})) == "other"


def test_middle_only_is_other():
    assert recognize_gesture(_hand({"middle"})) == "other"


def test_missing_landmarks_is_other():
    assert recognize_gesture(Hand(id=1, bbox=(0, 0, 10, 10), landmarks=[])) == "other"


def test_orientation_down_still_recognized():
    assert recognize_gesture(_hand({"index"}, pointing="down")) == "one"
    assert recognize_gesture(_hand(set(), pointing="down")) == "fist"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gesture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dishcounter.gesture'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/dishcounter/gesture.py`:

```python
"""Recognize simple hand gestures from MediaPipe landmarks. A finger is
'extended' when its tip is farther from the wrist than its mid-joint, which is
orientation-tolerant and needs no calibration. Pure geometry, no frame access."""

from __future__ import annotations

import math

_WRIST = 0
# finger name -> (tip landmark, pip landmark)
_FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}


def _extended_fingers(landmarks: list[tuple[float, float]]) -> set[str]:
    wrist = landmarks[_WRIST]
    out: set[str] = set()
    for name, (tip, pip) in _FINGERS.items():
        if math.dist(landmarks[tip], wrist) > math.dist(landmarks[pip], wrist):
            out.add(name)
    return out


def recognize_gesture(hand) -> str:
    lm = hand.landmarks
    if not lm or len(lm) < 21:
        return "other"
    ext = _extended_fingers(lm)
    if not ext:
        return "fist"
    if ext == {"index"}:
        return "one"
    if ext == {"index", "middle"}:
        return "two"
    return "other"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gesture.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/gesture.py tests/test_gesture.py
git commit -m "feat: recognize one/two/fist gestures from hand landmarks"
```

---

### Task 2: `session.py` — debounced `SessionController`

**Files:**
- Create: `src/dishcounter/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: gesture strings (`"one"`/`"two"`/`"fist"`/`"other"`) and a monotonic-ish `now` float.
- Produces: `SessionController(hold_seconds=1.0)` with `.active: str | None` and
  `update(gesture: str, now: float) -> str | None` (returns the active washer).

- [ ] **Step 1: Write the failing test**

Create `tests/test_session.py`:

```python
from dishcounter.session import SessionController


def test_gesture_held_long_enough_sets_washer():
    s = SessionController(hold_seconds=1.0)
    assert s.update("one", 0.0) is None       # just started holding
    assert s.update("one", 0.5) is None        # not held long enough
    assert s.update("one", 1.0) == "You"       # held 1.0s -> acts
    assert s.active == "You"


def test_two_fingers_sets_wife_and_fist_ends():
    s = SessionController(hold_seconds=1.0)
    s.update("two", 0.0)
    assert s.update("two", 1.0) == "Wife"
    s.update("fist", 1.0)
    assert s.update("fist", 2.0) is None       # fist held 1.0s -> session ends
    assert s.active is None


def test_brief_gesture_does_not_act():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    s.update("other", 0.3)                      # dropped before the hold elapsed
    assert s.update("one", 0.6) is None         # timer restarted at 0.6
    assert s.active is None


def test_changing_gesture_resets_the_timer():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    s.update("two", 0.5)                         # switched candidate
    assert s.update("two", 1.0) is None          # only 0.5s on 'two'
    assert s.update("two", 1.5) == "Wife"


def test_same_gesture_acts_once_then_is_idempotent():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("one", 1.0) == "You"
    # Still holding 'one' later must not re-trigger anything new.
    assert s.update("one", 5.0) == "You"
    assert s.active == "You"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dishcounter.session'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/dishcounter/session.py`:

```python
"""Turns debounced hand gestures into the active washer for a session. A
gesture must be held continuously for `hold_seconds` before it takes effect, so
a fleeting pose mid-wash will not flip the session."""

from __future__ import annotations


class SessionController:
    def __init__(self, hold_seconds: float = 1.0) -> None:
        self._hold = hold_seconds
        self.active: str | None = None
        self._candidate: str | None = None
        self._since: float | None = None
        self._acted = False

    def update(self, gesture: str, now: float) -> str | None:
        if gesture not in ("one", "two", "fist"):
            # 'other' / unrecognized breaks the hold streak.
            self._candidate = None
            self._since = None
            self._acted = False
            return self.active

        if gesture != self._candidate:
            self._candidate = gesture
            self._since = now
            self._acted = False

        if not self._acted and self._since is not None and (now - self._since) >= self._hold:
            self._acted = True
            if gesture == "one":
                self.active = "You"
            elif gesture == "two":
                self.active = "Wife"
            else:  # fist
                self.active = None
        return self.active
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_session.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/dishcounter/session.py tests/test_session.py
git commit -m "feat: add debounced SessionController for gesture-driven washer state"
```

---

### Task 3: Cutover logic — FusionEngine + Engine use the active washer

**Files:**
- Modify: `src/dishcounter/config.py` (add `gesture_hold`, additive)
- Modify: `src/dishcounter/fusion.py`
- Modify: `src/dishcounter/engine.py`
- Test: `tests/test_fusion.py` (rewrite), `tests/test_engine_e2e.py` (rewrite),
  `tests/test_annotate.py` (rewrite)

**Interfaces:**
- Consumes: `recognize_gesture` (Task 1), `SessionController` (Task 2), `Dish`, `Zone`, `WashEvent`.
- Produces:
  - `FusionEngine(sink, exit_grace=1.5, cooldown=3.0)` with
    `process(dishes, now, active_washer) -> list[WashEvent]`.
  - `annotate(frame, config, hands, dishes, active_washer, gesture) -> np.ndarray`.
  - `Thresholds.gesture_hold: float = 1.0`.

This task keeps `you_profile`/`wife_profile` in the config and the
`identity.py`/`ring.py` files in place (removed in Task 4) so the suite stays
green. `FusionEngine.process` drops the unused `hands` parameter (the engine
resolves gestures itself), an intentional simplification of the spec's
indicative signature.

- [ ] **Step 1: Add `gesture_hold` to config (additive)**

In `src/dishcounter/config.py`, add one field to `Thresholds` (keep all existing
fields including `identity_distance` for now):

```python
class Thresholds(BaseModel):
    identity_distance: float = 25.0  # (unused after gesture cutover; removed in cleanup)
    exit_grace: float = 1.5          # seconds a dish must be gone before counting
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act
```

- [ ] **Step 2: Write the failing FusionEngine test (rewrite `tests/test_fusion.py`)**

Replace the entire contents of `tests/test_fusion.py`:

```python
from dishcounter.config import Zone
from dishcounter.domain import Dish
from dishcounter.fusion import FusionEngine

SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _dish(cx, did, in_sink=True):
    cy = 50 if in_sink else 200
    return Dish(id=did, bbox=(cx - 10, cy - 10, cx + 10, cy + 10), label="plate",
                confidence=0.9)


def test_dish_washed_during_a_session_counts_for_active_washer():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    assert eng.process([_dish(50, 1)], now=0.0, active_washer="You") == []
    events = eng.process([], now=2.0, active_washer="You")  # gone > grace
    assert len(events) == 1
    assert events[0].person == "You"
    assert events[0].confidence == 1.0
    assert events[0].source_id == 1


def test_dish_washed_with_no_active_session_is_uncertain():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1)], now=0.0, active_washer=None)
    events = eng.process([], now=2.0, active_washer=None)
    assert len(events) == 1
    assert events[0].person == "uncertain"


def test_switching_active_washer_attributes_each_dish_correctly():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(40, 1)], now=0.0, active_washer="You")
    first = eng.process([], now=2.0, active_washer="You")          # You's dish exits
    eng.process([_dish(60, 2)], now=6.0, active_washer="Wife")     # new dish, Wife active
    second = eng.process([], now=8.0, active_washer="Wife")
    assert [e.person for e in first] == ["You"]
    assert [e.person for e in second] == ["Wife"]


def test_dish_never_in_sink_does_not_count():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1, in_sink=False)], now=0.0, active_washer="You")
    assert eng.process([], now=2.0, active_washer="You") == []


def test_two_dishes_same_session_within_cooldown_collapse_to_one():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(40, 1), _dish(60, 2)], now=0.0, active_washer="You")
    events = eng.process([], now=2.0, active_washer="You")
    assert len(events) == 1


def test_gone_exactly_at_grace_does_not_fire_but_just_past_does():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1)], now=0.0, active_washer="You")
    assert eng.process([], now=1.5, active_washer="You") == []     # == grace, no fire
    assert len(eng.process([], now=1.6, active_washer="You")) == 1  # just past
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `FusionEngine.__init__` still requires `identity`; `process` signature mismatch.

- [ ] **Step 4: Rewrite `src/dishcounter/fusion.py`**

Replace the entire contents of `src/dishcounter/fusion.py`:

```python
"""The counting rule. Dishes drive the count: a dish washed at the sink during
an active session, that then disappears from view, is one washed dish credited
to that session's washer. Identity comes from the session, not appearance."""

from __future__ import annotations

from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Dish, WashEvent


@dataclass
class _DishState:
    last_sink_time: float | None = None
    last_seen: float = 0.0
    locked_person: str = "uncertain"
    locked_confidence: float = 0.0


class FusionEngine:
    def __init__(self, sink: Zone, exit_grace: float = 1.5, cooldown: float = 3.0) -> None:
        self._sink = sink
        self._exit_grace = exit_grace
        self._cooldown = cooldown
        self._states: dict[int, _DishState] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self, dishes: list[Dish], now: float, active_washer: str | None
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
                if state.locked_person == "uncertain" and active_washer is not None:
                    state.locked_person = active_washer
                    state.locked_confidence = 1.0

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

    def _can_fire(self, person: str, now: float) -> bool:
        if person not in ("You", "Wife"):
            return True  # uncertain: recorded, not counted, no cooldown
        last = self._last_person_fire.get(person)
        return last is None or (now - last) >= self._cooldown
```

- [ ] **Step 5: Run the FusionEngine test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_fusion.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Rewrite the Engine wiring and overlay (`src/dishcounter/engine.py`)**

Replace the imports block, `annotate`, the `Engine.__init__` body, and
`process_frame`. First, the imports (drop `IdentityClassifier`, `ring_metrics`,
`median_chroma`; add `recognize_gesture`, `SessionController`):

```python
from dishcounter.config import Config
from dishcounter.dish_detector import DishDetector
from dishcounter.domain import WashEvent
from dishcounter.fusion import FusionEngine
from dishcounter.gesture import recognize_gesture
from dishcounter.session import SessionController
from dishcounter.state import SharedState
from dishcounter.store import CountStore
from dishcounter.tracker import IouTracker
```

Replace `annotate` with:

```python
def annotate(frame: np.ndarray, config: Config, hands, dishes,
             active_washer, gesture) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    z = config.sink_zone
    cv2.rectangle(out, (z.x1, z.y1), (z.x2, z.y2), (255, 0, 0), 2)
    for dish in dishes:
        x1, y1, x2, y2 = dish.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, dish.label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
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

In `Engine.__init__`, replace the tracker/identity/fusion block (the lines that
build `self._identity` and `self._fusion`) with:

```python
        t = config.thresholds
        self._hand_tracker = IouTracker(iou_threshold=t.iou_match)
        self._dish_tracker = IouTracker(iou_threshold=t.iou_match)
        self._session = SessionController(hold_seconds=t.gesture_hold)
        self._fusion = FusionEngine(
            config.sink_zone, exit_grace=t.exit_grace, cooldown=t.cooldown
        )
```

Replace `process_frame` and add the gesture resolver:

```python
    @staticmethod
    def _resolve_gesture(hands) -> str:
        for hand in hands:
            g = recognize_gesture(hand)
            if g in ("one", "two", "fist"):
                return g
        return "other"

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame))
        dishes = self._dish_tracker.update(self._dish_detector.detect(frame))
        gesture = self._resolve_gesture(hands)
        active = self._session.update(gesture, now)
        events = self._fusion.process(dishes, now, active)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands, dishes, active, gesture)
        self._state.publish(
            self._encode(annotated), self._store.totals(now), camera_online=True
        )
        return events
```

(The default `annotator: Callable = annotate` in `__init__` is unchanged.)

- [ ] **Step 7: Rewrite the overlay test (`tests/test_annotate.py`)**

Replace the entire contents of `tests/test_annotate.py`:

```python
import numpy as np

from dishcounter.config import Config, Zone
from dishcounter.domain import Dish, Hand
from dishcounter.engine import annotate


def _cfg() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=50, y2=50),
        you_profile=__import__("dishcounter.config", fromlist=["SkinProfile"])
        .SkinProfile(cr=165.0, cb=110.0),
        wife_profile=__import__("dishcounter.config", fromlist=["SkinProfile"])
        .SkinProfile(cr=120.0, cb=150.0),
    )


def test_annotate_runs_with_active_session_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]
    dishes = [Dish(id=1, bbox=(5, 5, 40, 40), label="plate", confidence=0.9)]
    out = annotate(frame, _cfg(), hands, dishes, "You", "one")
    assert out.shape == frame.shape
    assert out.any()  # something was drawn on the all-zero frame


def test_annotate_runs_with_no_session():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [], [], None, "other")
    assert out.shape == frame.shape
```

(`_cfg` keeps the skin-profile fields because `Config` still requires them in
this task; Task 4 removes both the fields and this awkward import.)

- [ ] **Step 8: Rewrite the end-to-end test (`tests/test_engine_e2e.py`)**

Replace the entire contents of `tests/test_engine_e2e.py`:

```python
import numpy as np

from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.dish_detector import FakeDishDetector
from dishcounter.domain import Dish, Hand
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


def _hand(extended):
    return Hand(id=None, bbox=(150, 150, 190, 190), landmarks=_landmarks(extended))


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


def test_session_started_with_one_finger_counts_dish_for_you(sample_config, blank_frame):
    one = _hand({"index"})
    engine = _engine(
        sample_config,
        # hold 'one' across two frames to clear the 1.0s gesture_hold, then idle.
        hand_script=[[one], [one], [one], []],
        dish_script=[[], [], [_dish(50)], []],   # dish appears in sink once session active
        frames=[blank_frame] * 4,
        times=[0.0, 1.0, 1.2, 3.5],
    )
    engine.run()  # FakeCamera exhausts after 4 frames
    assert engine._store.totals(3.5)["all_time"] == {"You": 1, "Wife": 0}


def test_dish_washed_with_no_session_is_not_counted(sample_config, blank_frame):
    engine = _engine(
        sample_config,
        hand_script=[[], []],
        dish_script=[[_dish(50)], []],
        frames=[blank_frame, blank_frame],
        times=[0.0, 2.0],
    )
    engine.run()
    assert engine._store.totals(2.0)["all_time"] == {"You": 0, "Wife": 0}
```

- [ ] **Step 9: Run the full suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all tests; `test_identity.py` and `test_ring.py` still pass —
those files are removed in Task 4).

- [ ] **Step 10: Commit**

```bash
git add src/dishcounter/config.py src/dishcounter/fusion.py src/dishcounter/engine.py \
        tests/test_fusion.py tests/test_engine_e2e.py tests/test_annotate.py
git commit -m "feat: attribute dishes by gesture-driven session instead of skin tone"
```

---

### Task 4: Cleanup — remove the skin-tone subsystem and skin calibration

**Files:**
- Modify: `src/dishcounter/config.py` (remove `you_profile`, `wife_profile`, `identity_distance`)
- Modify: `tests/conftest.py`, `tests/test_config.py`
- Modify: `src/dishcounter/calibrate.py` (sink-zone only), `tests/test_calibrate.py`
- Modify: `tests/test_annotate.py` (drop the skin-profile import hack from Task 3)
- Delete: `src/dishcounter/identity.py`, `tests/test_identity.py`,
  `src/dishcounter/ring.py`, `tests/test_ring.py`

**Interfaces:**
- Produces: `Config` without skin profiles; `Thresholds` without `identity_distance`.

- [ ] **Step 1: Write/adjust the failing test**

In `tests/test_config.py`, replace any test that constructs a `Config` with
skin profiles, and assert the fields are gone. Add:

```python
def test_config_has_no_skin_profiles_and_no_identity_distance():
    from dishcounter.config import Config, Zone

    cfg = Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=10, y2=10))
    assert not hasattr(cfg, "you_profile")
    assert not hasattr(cfg, "wife_profile")
    assert not hasattr(cfg.thresholds, "identity_distance")
    assert cfg.thresholds.gesture_hold == 1.0
```

Update `tests/conftest.py` `sample_config` to drop the profile fields and the
`identity_distance` argument:

```python
@pytest.fixture
def sample_config() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        thresholds=Thresholds(gesture_hold=1.0),
    )
```

(Remove the now-unused `SkinProfile` import from `conftest.py`.)

Update `tests/test_annotate.py` `_cfg` to drop the skin profiles:

```python
def _cfg() -> Config:
    return Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=50, y2=50))
```

(Remove the `__import__(... SkinProfile ...)` lines.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `Config` still requires `you_profile`/`wife_profile`; `identity_distance` still present.

- [ ] **Step 3: Remove skin profiles from config**

In `src/dishcounter/config.py`:
- Delete the `SkinProfile` class **only if** nothing else imports it — `calibrate.py`
  is updated in this task to stop using it, so after Step 4 it is unused; delete it.
- Remove `identity_distance` from `Thresholds`.
- Remove `you_profile` and `wife_profile` from `Config`.

Resulting `Thresholds` and `Config`:

```python
class Thresholds(BaseModel):
    exit_grace: float = 1.5          # seconds a dish must be gone before counting
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act


class Config(BaseModel):
    camera_index: int
    sink_zone: Zone
    dish_classes: list[str] = DEFAULT_DISH_CLASSES
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"
    thresholds: Thresholds = Thresholds()
```

- [ ] **Step 4: Make calibration sink-zone only**

Replace the contents of `src/dishcounter/calibrate.py` with a sink-only flow
(no skin capture, no `SkinProfile`/`median_chroma`/`profile_*` helpers):

```python
"""One-time (re-runnable) calibration: drag the sink rectangle. Persists only
the sink zone — never raw images. Identity is by gesture, not skin tone."""

from __future__ import annotations

from pathlib import Path

from dishcounter.config import Config, Thresholds, Zone


def zone_from_drag(start: tuple[int, int], end: tuple[int, int]) -> Zone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return Zone(x1=x1, y1=y1, x2=x2, y2=y2)


def run_calibration(
    config_path: str | Path = "config.yaml", camera_index: int = 0
) -> None:  # pragma: no cover - interactive, exercised manually
    import cv2  # noqa: PLC0415

    from dishcounter.camera import Camera

    cam = Camera(camera_index)
    print("Calibration: drag a rectangle around the sink, then press ENTER.")
    frame = None
    while frame is None:
        frame = cam.read()
    roi = cv2.selectROI("drag the sink zone", frame, showCrosshair=True)
    x, y, bw, bh = (int(v) for v in roi)
    sink_zone = zone_from_drag((x, y), (x + bw, y + bh))
    cv2.destroyAllWindows()
    cam.release()

    config = Config(
        camera_index=camera_index, sink_zone=sink_zone, thresholds=Thresholds()
    )
    config.save(config_path)
    print(f"Saved calibration to {config_path}.")
```

Replace `tests/test_calibrate.py` entirely (the old skin-profile/guard tests no
longer apply):

```python
from dishcounter.calibrate import zone_from_drag


def test_zone_from_drag_normalizes_corners():
    zone = zone_from_drag((120, 90), (20, 10))  # dragged up-left
    assert zone.x1 == 20
    assert zone.y1 == 10
    assert zone.x2 == 120
    assert zone.y2 == 90
    assert zone.contains((70, 50))


def test_calibration_module_does_not_reference_skin_or_drying():
    import inspect

    from dishcounter import calibrate

    src = inspect.getsource(calibrate)
    assert "drying" not in src
    assert "SkinProfile" not in src
    assert "profile" not in src
```

- [ ] **Step 5: Delete the dead modules and their tests**

```bash
git rm src/dishcounter/identity.py tests/test_identity.py \
       src/dishcounter/ring.py tests/test_ring.py
```

- [ ] **Step 6: Run the full suite and a grep to verify**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

Run: `grep -rn "IdentityClassifier\|you_profile\|wife_profile\|identity_distance\|ring_metrics\|from dishcounter.ring\|from dishcounter.identity" src/ tests/`
Expected: (no matches)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove skin-tone identity, ring probe, and skin calibration"
```

(`git add -A` is acceptable here only because the working tree is clean apart
from this task's changes; confirm with `git status` before committing.)

---

### Task 5: Cleanup — remove the now-unused `Hand.region_pixels` skin sample

**Files:**
- Modify: `src/dishcounter/domain.py` (remove `region_pixels` from `Hand`)
- Modify: `src/dishcounter/detector.py` (stop sampling the skin region)
- Test: `tests/test_domain.py` (if it references `region_pixels`)

**Interfaces:**
- Produces: `Hand(id, bbox, landmarks=[], confidence=0.0)` — no `region_pixels`.

- [ ] **Step 1: Confirm no remaining consumers**

Run: `grep -rn "region_pixels" src/ tests/`
Expected (before this task): matches only in `domain.py` (field), `detector.py`
(sampling), and possibly `tests/test_domain.py`. If any other test still sets it,
that test belongs to an earlier task that should already have dropped it —
update it here.

- [ ] **Step 2: Write/adjust the failing test**

In `tests/test_domain.py`, ensure there is a test asserting the field is gone
(add it; remove any construction that passes `region_pixels`):

```python
def test_hand_has_no_region_pixels_field():
    from dishcounter.domain import Hand

    hand = Hand(id=1, bbox=(0, 0, 10, 10))
    assert not hasattr(hand, "region_pixels")
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_domain.py -v`
Expected: FAIL — `Hand` still has `region_pixels`.

- [ ] **Step 4: Remove the field and its sampling**

In `src/dishcounter/domain.py`, remove the `region_pixels` field from `Hand` (and
drop the now-unneeded `import numpy as np` only if nothing else in the file uses
it — `median_chroma`/`bgr_to_ycrcb` still use numpy, so keep the import).

Resulting `Hand`:

```python
@dataclass
class Hand:
    """A detected hand for one frame."""

    id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
```

In `src/dishcounter/detector.py`:
- Delete the `_sample_region` method on `MediaPipeHandDetector`.
- In the `detect` loop, remove the `region = self._sample_region(...)` line and the
  `region_pixels=region` argument when constructing `Hand`. Remove the now-unused
  `region_size` constructor parameter and `self._region_size`.

The `Hand(...)` construction in `detect` becomes:

```python
            hands.append(
                Hand(
                    id=None,
                    bbox=bbox,
                    landmarks=pts,
                    confidence=score,
                )
            )
```

- [ ] **Step 5: Run the full suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dishcounter/domain.py src/dishcounter/detector.py tests/test_domain.py
git commit -m "refactor: drop unused Hand.region_pixels skin sample"
```

---

### Task 6: Update the README for gesture identity

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Rewrite the affected sections**

Update `README.md` so it describes gesture identity instead of skin tone:
- Intro bullets: identity is by a start/end **hand gesture** (1 = You, 2 = Wife,
  fist = end), not skin tone.
- Section 3 (Calibrate): now only drags the sink zone — remove the `y`/`w`
  skin-capture steps.
- Section 4 (Run): explain the workflow — show 1 or 2 fingers to start your
  session (hold ~1s), wash, show a fist to end; the dashboard banner shows the
  active washer and the live gesture; nothing counts with no active session.
- Section 6 (Tuning): replace `identity_distance` with `gesture_hold`
  (seconds a gesture must be held; raise if sessions start too easily, lower for
  snappier switching). Keep `exit_grace`, `cooldown`, `iou_match`, dish settings.
- Section 7 (How it works): update the pipeline diagram — hands feed gesture
  recognition → session controller; dishes feed the fusion engine; identity is
  the active washer.
- Section 8 (Troubleshooting): replace skin-tone rows with gesture ones
  (e.g. "session won't start → hold 1/2 fingers steady for ~1s facing the
  camera, watch the banner"; "wrong person counting → show the right finger
  count again, or fist then restart").
- Privacy: `config.yaml` now stores only the sink rectangle and settings (no
  skin profiles).

- [ ] **Step 2: Verify the suite still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document gesture-based session identity"
```

---

## Self-Review

**Spec coverage:**
- Gesture recognition (one/two/fist/other from landmarks, wrist-distance rule) → Task 1.
- Debounced session controller (hold to act, idempotent, fist ends) → Task 2.
- FusionEngine attributes by active washer; uncertain with no session; confidence 1.0 → Task 3.
- Engine resolves one gesture/frame, drives SessionController, passes active washer; overlay banner + gesture → Task 3.
- `gesture_hold` config; remove skin profiles/`identity_distance`; sink-only calibration → Tasks 3 (add) + 4 (remove).
- Remove `identity.py`, `ring.py`, and their tests → Task 4.
- Remove `Hand.region_pixels` + skin sampling → Task 5.
- README → Task 6.

**Placeholder scan:** none — every code/test step has complete code; commands have expected output.

**Type consistency:** `recognize_gesture(hand) -> str`, `SessionController(hold_seconds).update(gesture, now) -> str | None` / `.active`, `FusionEngine(sink, exit_grace, cooldown).process(dishes, now, active_washer)`, `annotate(frame, config, hands, dishes, active_washer, gesture)`, and `Engine._resolve_gesture(hands)` are used identically across the tasks that define and consume them. The cutover (Task 3) keeps `Config` skin fields so the suite stays green; Task 4 removes them together with their last consumers (`conftest`, `test_annotate` hack).
