# Rackwash Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a **new, fully self-contained package `rackwash`** (own folder, own `pyproject.toml`, zero dependency on `dishcounter`) that counts dish-washing by the net increase of dishware in the drying racks between gesture-marked session boundaries.

**Architecture:** A new `rackwash/` project mirrors the proven infrastructure of `dishcounter` (camera, hand/dish detectors, IoU tracker, gestures, session controller, SQLite store, thread-safe state, FastAPI dashboard) by **copying** that reviewed code and re-namespacing it. The only genuinely new logic is the rack-delta counting model: rack-region config, `RackDeltaCounter`, and an engine that runs YOLO bursts only at session boundaries. `dishcounter` is never modified or imported.

**Tech Stack:** Python 3.12, MediaPipe (hands/gestures), Ultralytics YOLO-World (dishware + person), OpenCV, FastAPI, Pydantic, SQLite, pytest, ruff.

## Global Constraints

- **Do not modify or import `dishcounter`.** `rackwash` is standalone; after copying, every import inside `rackwash` is `from rackwash.…`.
- Build `rackwash/` as its own project (own `pyproject.toml`, `src/rackwash/` layout, `tests/`), installed into the existing `.venv` via `pip install -e ./rackwash` (reuses the already-downloaded MediaPipe/YOLO wheels).
- Run commands via the venv: `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `.venv/bin/python`. Run rackwash tests with `.venv/bin/python -m pytest rackwash/tests -q` (uses `rackwash/pyproject.toml` as the config). Run ruff on rackwash with `.venv/bin/ruff check rackwash`.
- YOLO is a **counter of dishware in a static rack region**, never a tracker. It runs **only during boundary bursts** (the `RackDeltaCounter` calls the injected `detect` callback only while collecting).
- A session's count = `Σ_rack max(0, end[rack] − baseline[rack])`, per-rack clamped, only for racks measured at **both** boundaries (else that rack contributes 0). Emit that many identical `WashEvent`s for the closing washer (`source_id = -1`).
- Gestures register **anywhere** (no sink zone); a hand showing `one`(You)/`two`(Wife) held `gesture_hold` = **1.5s** toggles that person.
- A rack with `requires_clear=True` is sampled only on frames where **no `person` detection and no tracked hand overlaps that rack's rectangle**; `requires_clear=False` racks are always sampled.
- `dish_classes` = `["plate","bowl","cup","glass","mug"]` (no cutlery). The CLI builds YOLO with classes `dish_classes + ["person"]`.
- Bias toward **undercounting**: clamp deltas at ≥0; an unmeasurable rack contributes 0. Labels exactly `"You"`/`"Wife"`. No `ultralytics` import in tests.
- `git add` only the files each task names — never `git add -A`/`.` (gitignored scratch under `.superpowers/`).
- End every commit message with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

```
rackwash/
  pyproject.toml                 # own project metadata + deps + console script   (T1)
  README.md                                                                        (T10)
  src/rackwash/
    __init__.py                                                                    (T1)
    domain.py        # COPY: Hand, WashEvent, Dish                                 (T2)
    camera.py        # COPY: Camera, FakeCamera                                    (T2)
    gesture.py       # COPY: recognize_gesture                                     (T2)
    session.py       # COPY: SessionController                                     (T2)
    tracker.py       # COPY: IouTracker                                            (T2)
    state.py         # COPY: SharedState                                           (T2)
    store.py         # COPY: CountStore                                            (T3)
    detector.py      # COPY: HandDetector/Fake/MediaPipe                           (T3)
    dish_detector.py # COPY: DishDetector/Fake/YoloWorld                           (T3)
    dashboard.py     # COPY: FastAPI app                                           (T3)
    config.py        # NEW: RackZone + Zone.overlaps, Config(rack_zones)           (T4)
    rack.py          # NEW: count_dishware, rack_occluded                          (T5)
    rack_delta.py    # NEW: RackDeltaCounter (the heart)                           (T6)
    engine.py        # NEW: Engine + annotate (rack-delta wiring)                  (T7)
    calibrate.py     # NEW: draw rack zones + requires_clear                       (T8)
    cli.py           # NEW: rackwash calibrate / run                               (T9)
  tests/
    test_domain.py test_camera.py test_gesture.py test_session.py
    test_tracker.py test_state.py                          # COPY (re-namespaced)  (T2)
    test_store.py test_detector.py test_dish_detector.py test_dashboard.py  # COPY (T3)
    test_config.py        # NEW                                                    (T4)
    test_rack.py          # NEW                                                    (T5)
    test_rack_delta.py    # NEW                                                    (T6)
    conftest.py test_engine_e2e.py test_annotate.py  # NEW                         (T7)
    test_calibrate.py     # NEW                                                    (T8)
    test_cli.py           # NEW                                                    (T9)
```

---

### Task 1: Scaffold the `rackwash` project

**Files:**
- Create: `rackwash/pyproject.toml`, `rackwash/src/rackwash/__init__.py`, `rackwash/tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `rackwash` package in `.venv` with a `rackwash` console script (→ `rackwash.cli:main`), working `pytest`/`ruff`.

- [ ] **Step 1: Write `rackwash/pyproject.toml`**

```toml
[project]
name = "rackwash"
version = "0.1.0"
description = "Outcome-based dish-washing scoreboard (rack-delta counting)"
requires-python = ">=3.12,<3.13"
dependencies = [
    "opencv-python>=4.9",
    "mediapipe>=0.10",
    "numpy>=1.26",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "websockets>=12",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "ultralytics>=8.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4", "httpx>=0.27"]

[project.scripts]
rackwash = "rackwash.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = [
    "ignore:Using `httpx` with `starlette.testclient` is deprecated:Warning",
]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
```

- [ ] **Step 2: Create the package skeleton**

`rackwash/src/rackwash/__init__.py`:
```python
"""Outcome-based dish-washing scoreboard: count by rack delta per session."""

__version__ = "0.1.0"
```

`rackwash/tests/__init__.py`:
```python
```

- [ ] **Step 3: Install into the existing venv and verify**

Run:
```bash
.venv/bin/python -m pip install -e ./rackwash
.venv/bin/python -c "import rackwash; print(rackwash.__version__)"
.venv/bin/python -m pytest rackwash/tests -q
.venv/bin/ruff check rackwash
```
Expected: prints `0.1.0`; pytest reports `no tests ran`; ruff `All checks passed!`. (`dishcounter` remains installed and untouched.)

- [ ] **Step 4: Commit**

```bash
git add rackwash/pyproject.toml rackwash/src/rackwash/__init__.py rackwash/tests/__init__.py
git commit -m "chore: scaffold standalone rackwash package

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Copy the leaf infrastructure modules

**Files:**
- Create (copy of `dishcounter`): `rackwash/src/rackwash/{domain,camera,gesture,session,tracker,state}.py`
- Create (copy of `dishcounter`): `rackwash/tests/test_{domain,camera,gesture,session,tracker,state}.py`

**Interfaces:**
- Consumes: nothing (these have no internal imports).
- Produces: `Hand`, `WashEvent`, `Dish` (domain); `Camera`/`FakeCamera`; `recognize_gesture`; `SessionController(hold_seconds=…)`; `IouTracker(iou_threshold=…, coast_seconds=…)`; `SharedState`.

- [ ] **Step 1: Copy each module verbatim with re-namespaced imports**

For each of `domain`, `camera`, `gesture`, `session`, `tracker`, `state`:
- Copy `src/dishcounter/<name>.py` → `rackwash/src/rackwash/<name>.py`.
- Copy `tests/test_<name>.py` → `rackwash/tests/test_<name>.py`.
- In **both** copies, replace every occurrence of `from dishcounter.` with `from rackwash.` and every `import dishcounter` with `import rackwash`.

(These six source modules have no internal `dishcounter` imports; only their test files import `from dishcounter.<name> import …`, which becomes `from rackwash.<name> import …`. Read each source/test file to confirm before copying — do not hand-retype; copy the real content.)

- [ ] **Step 2: Run the copied tests**

Run: `.venv/bin/python -m pytest rackwash/tests -q`
Expected: PASS — all copied leaf-module tests pass under the `rackwash` namespace.

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check rackwash/src/rackwash rackwash/tests`
Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add rackwash/src/rackwash/domain.py rackwash/src/rackwash/camera.py rackwash/src/rackwash/gesture.py rackwash/src/rackwash/session.py rackwash/src/rackwash/tracker.py rackwash/src/rackwash/state.py rackwash/tests/test_domain.py rackwash/tests/test_camera.py rackwash/tests/test_gesture.py rackwash/tests/test_session.py rackwash/tests/test_tracker.py rackwash/tests/test_state.py
git commit -m "chore: copy leaf infra (domain, camera, gesture, session, tracker, state)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Copy the detector, persistence, and dashboard modules

**Files:**
- Create (copy of `dishcounter`): `rackwash/src/rackwash/{store,detector,dish_detector,dashboard}.py`
- Create (copy of `dishcounter`): `rackwash/tests/test_{store,detector,dish_detector,dashboard}.py`

**Interfaces:**
- Consumes: `WashEvent`, `Hand`, `Dish` (domain, T2), `SharedState` (T2).
- Produces: `CountStore(db_path)` with `record`/`totals`; `HandDetector`/`FakeHandDetector`/`MediaPipeHandDetector`; `DishDetector`/`FakeDishDetector`/`dishes_from_detections`/`YoloWorldDishDetector`; `create_app(state)`/`run_server(state, host, port)`.

- [ ] **Step 1: Copy each module verbatim with re-namespaced imports**

For each of `store`, `detector`, `dish_detector`, `dashboard`:
- Copy `src/dishcounter/<name>.py` → `rackwash/src/rackwash/<name>.py`.
- Copy `tests/test_<name>.py` → `rackwash/tests/test_<name>.py`.
- In both copies, replace every `from dishcounter.` with `from rackwash.` (e.g. `store.py`'s `from dishcounter.domain import WashEvent` → `from rackwash.domain import WashEvent`; `dashboard.py`'s `from dishcounter.state import SharedState` → `from rackwash.state import SharedState`; `detector.py`'s `from dishcounter.domain import Hand`; `dish_detector.py`'s `from dishcounter.domain import Dish`).

Read each file first and copy real content (these contain lazy `# noqa: PLC0415` imports and `# pragma: no cover` markers that must be preserved).

- [ ] **Step 2: Run the copied tests**

Run: `.venv/bin/python -m pytest rackwash/tests -q`
Expected: PASS — all copied module tests pass (including `test_dashboard.py`, whose Starlette deprecation warning is filtered by `rackwash/pyproject.toml`).

- [ ] **Step 3: Lint**

Run: `.venv/bin/ruff check rackwash/src/rackwash rackwash/tests`
Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add rackwash/src/rackwash/store.py rackwash/src/rackwash/detector.py rackwash/src/rackwash/dish_detector.py rackwash/src/rackwash/dashboard.py rackwash/tests/test_store.py rackwash/tests/test_detector.py rackwash/tests/test_dish_detector.py rackwash/tests/test_dashboard.py
git commit -m "chore: copy store, detector, dish_detector, dashboard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `config.py` — `RackZone`, `Zone.overlaps`, rack-region `Config`

**Files:**
- Create: `rackwash/src/rackwash/config.py`
- Test: `rackwash/tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Zone(x1,y1,x2,y2)` with `contains(point)` and `overlaps(bbox)`.
  - `RackZone(Zone)` with `requires_clear: bool = True`.
  - `Thresholds(iou_match=0.3, gesture_hold=1.5, track_coast=2.0, rack_window=1.5, dish_interval=0.5)`.
  - `Config(camera_index, rack_zones: list[RackZone] = [], thresholds=Thresholds(), dish_classes=[…dishware…], dish_conf=0.4, yolo_model="yolov8s-worldv2.pt")` with `load(path)` (raises `FileNotFoundError` mentioning `rackwash calibrate`) and `save(path)`.

- [ ] **Step 1: Write the failing test**

Create `rackwash/tests/test_config.py`:
```python
import pytest

from rackwash.config import Config, RackZone, Thresholds, Zone


def test_zone_contains_and_overlaps():
    z = Zone(x1=0, y1=0, x2=100, y2=100)
    assert z.contains((50, 50)) is True
    assert z.contains((150, 50)) is False
    assert z.overlaps((50, 50, 150, 150)) is True   # corner overlap
    assert z.overlaps((101, 0, 200, 100)) is False   # just right of zone
    assert z.overlaps((100, 100, 200, 200)) is True  # touching corner counts


def test_rack_zone_defaults_requires_clear_true():
    rz = RackZone(x1=0, y1=0, x2=10, y2=10)
    assert rz.requires_clear is True
    assert rz.contains((5, 5)) is True


def test_thresholds_defaults():
    t = Thresholds()
    assert t.gesture_hold == 1.5
    assert t.rack_window == 1.5
    assert t.dish_interval == 0.5
    assert t.iou_match == 0.3
    assert t.track_coast == 2.0


def test_config_round_trips_rack_zones(tmp_path):
    path = tmp_path / "config.yaml"
    Config(
        camera_index=0,
        rack_zones=[
            RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False),
            RackZone(x1=120, y1=0, x2=200, y2=80, requires_clear=True),
        ],
    ).save(path)
    loaded = Config.load(path)
    assert len(loaded.rack_zones) == 2
    assert loaded.rack_zones[0].requires_clear is False
    assert loaded.rack_zones[1].requires_clear is True
    assert "plate" in loaded.dish_classes and "fork" not in loaded.dish_classes


def test_load_missing_file_tells_user_to_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Config.load(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rackwash/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.config'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/config.py`**

```python
"""Pydantic config persisted to YAML. Written by `calibrate`, read by `run`.
Holds the drying-rack regions and tuning thresholds."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Zone(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        lo_x, hi_x = sorted((self.x1, self.x2))
        lo_y, hi_y = sorted((self.y1, self.y2))
        return lo_x <= x <= hi_x and lo_y <= y <= hi_y

    def overlaps(self, bbox: tuple[int, int, int, int]) -> bool:
        bx1, by1, bx2, by2 = bbox
        lo_x, hi_x = sorted((self.x1, self.x2))
        lo_y, hi_y = sorted((self.y1, self.y2))
        blo_x, bhi_x = sorted((bx1, bx2))
        blo_y, bhi_y = sorted((by1, by2))
        return not (bhi_x < lo_x or blo_x > hi_x or bhi_y < lo_y or blo_y > hi_y)


class RackZone(Zone):
    requires_clear: bool = True  # skip counting on frames where a person/hand overlaps it


class Thresholds(BaseModel):
    iou_match: float = 0.3       # IoU needed to keep a hand track's id
    gesture_hold: float = 1.5    # seconds a gesture must be held to act
    track_coast: float = 2.0     # seconds a lost hand track is kept alive
    rack_window: float = 1.5     # seconds a boundary counting burst lasts
    dish_interval: float = 0.5   # min seconds between dish-detector runs in a burst


class Config(BaseModel):
    camera_index: int
    rack_zones: list[RackZone] = Field(default_factory=list)
    thresholds: Thresholds = Thresholds()
    dish_classes: list[str] = Field(
        default_factory=lambda: ["plate", "bowl", "cup", "glass", "mug"]
    )
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No config at {path}. Run `rackwash calibrate` first."
            )
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rackwash/tests/test_config.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/config.py rackwash/tests/test_config.py
git commit -m "feat: rackwash config with RackZone, overlaps, rack regions

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: `rack.py` — pure counting/occlusion helpers

**Files:**
- Create: `rackwash/src/rackwash/rack.py`
- Test: `rackwash/tests/test_rack.py`

**Interfaces:**
- Consumes: `RackZone` (T4), `Dish`/`Hand` (T2).
- Produces:
  - `count_dishware(dishes: list[Dish], zone: RackZone, dishware_labels: set[str]) -> int`.
  - `rack_occluded(zone: RackZone, persons: list[Dish], hands: list[Hand]) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `rackwash/tests/test_rack.py`:
```python
from rackwash.config import RackZone
from rackwash.domain import Dish, Hand
from rackwash.rack import count_dishware, rack_occluded

LABELS = {"plate", "bowl", "cup", "glass", "mug"}
ZONE = RackZone(x1=0, y1=0, x2=100, y2=100)


def _dish(cx, cy, label="plate"):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label=label, confidence=0.9)


def test_count_dishware_counts_in_zone_dishware_only():
    dishes = [
        _dish(50, 50, "plate"),    # in zone, dishware -> counts
        _dish(50, 50, "fork"),     # not dishware -> ignored
        _dish(200, 50, "plate"),   # outside zone -> ignored
        _dish(50, 50, "person"),   # not dishware -> ignored
    ]
    assert count_dishware(dishes, ZONE, LABELS) == 1


def test_rack_occluded_true_when_person_box_overlaps():
    person = Dish(id=None, bbox=(80, 80, 160, 160), label="person", confidence=0.9)
    assert rack_occluded(ZONE, [person], []) is True


def test_rack_occluded_true_when_hand_box_overlaps():
    assert rack_occluded(ZONE, [], [Hand(id=1, bbox=(90, 90, 130, 130))]) is True


def test_rack_occluded_false_when_nothing_overlaps():
    person = Dish(id=None, bbox=(200, 200, 260, 260), label="person", confidence=0.9)
    assert rack_occluded(ZONE, [person], [Hand(id=1, bbox=(300, 300, 320, 320))]) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rackwash/tests/test_rack.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.rack'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/rack.py`**

```python
"""Pure helpers for counting dishware in a rack region and deciding whether a
rack is occluded. No model, no state."""

from __future__ import annotations

from rackwash.config import RackZone
from rackwash.domain import Dish, Hand


def count_dishware(dishes: list[Dish], zone: RackZone, dishware_labels: set[str]) -> int:
    return sum(
        1 for d in dishes if d.label in dishware_labels and zone.contains(d.centroid)
    )


def rack_occluded(zone: RackZone, persons: list[Dish], hands: list[Hand]) -> bool:
    return any(zone.overlaps(p.bbox) for p in persons) or any(
        zone.overlaps(h.bbox) for h in hands
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rackwash/tests/test_rack.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/rack.py rackwash/tests/test_rack.py
git commit -m "feat: rack-count and occlusion helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: `rack_delta.py` — `RackDeltaCounter` (the heart)

**Files:**
- Create: `rackwash/src/rackwash/rack_delta.py`
- Test: `rackwash/tests/test_rack_delta.py`

**Interfaces:**
- Consumes: `RackZone` (T4), `count_dishware`/`rack_occluded` (T5), `Dish`/`Hand`/`WashEvent` (T2).
- Produces:
  - `RackDeltaCounter(rack_zones: list[RackZone], dishware_labels: set[str], rack_window: float = 1.5, dish_interval: float = 0.5)`.
  - `process(active_washer: str | None, hands: list[Hand], now: float, detect: Callable[[], list[Dish]] | None) -> list[WashEvent]`.
  - `is_collecting` (read-only property).

- [ ] **Step 1: Write the failing test**

Create `rackwash/tests/test_rack_delta.py`:
```python
from rackwash.config import RackZone
from rackwash.domain import Dish
from rackwash.rack_delta import RackDeltaCounter

LABELS = {"plate", "bowl", "cup", "glass", "mug"}
R = RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False)
RC = RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=True)


def _plate(cx, cy=50):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label="plate", confidence=0.9)


def _det(dishes):
    return lambda: list(dishes)


def test_session_counts_dishes_added_to_rack():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    assert c.process("You", [], 1.0, _det([])) == []        # finalize start, baseline 0
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det(two))
    c.process(None, [], 10.5, _det(two))
    ev = c.process(None, [], 11.0, _det(two))
    assert len(ev) == 2 and all(e.person == "You" and e.source_id == -1 for e in ev)


def test_no_dishes_added_counts_zero():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    c.process(None, [], 10.0, _det([]))
    c.process(None, [], 10.5, _det([]))
    assert c.process(None, [], 11.0, _det([])) == []


def test_dishes_removed_counts_zero():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    two = [_plate(20), _plate(40)]
    c.process("You", [], 0.0, _det(two))
    c.process("You", [], 0.5, _det(two))
    c.process("You", [], 1.0, _det(two))
    c.process(None, [], 10.0, _det([]))
    c.process(None, [], 10.5, _det([]))
    assert c.process(None, [], 11.0, _det([])) == []


def test_burst_median_ignores_a_flicker_frame():
    c = RackDeltaCounter([R], LABELS, rack_window=1.5, dish_interval=0.4)
    for t in (0.0, 0.5, 1.0, 1.5):
        c.process("You", [], t, _det([]))
    two = [_plate(20), _plate(40)]
    three = [_plate(20), _plate(40), _plate(60)]
    c.process(None, [], 10.0, _det(two))
    c.process(None, [], 10.5, _det(two))
    c.process(None, [], 11.0, _det(three))   # flicker
    ev = c.process(None, [], 11.5, _det(two))
    assert len(ev) == 2


def test_requires_clear_skips_occluded_then_counts_clear_frame():
    c = RackDeltaCounter([RC], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    ev = c.process(None, [], 11.0, _det(two))   # clear frame -> counts
    assert len(ev) == 2


def test_requires_clear_all_occluded_counts_zero():
    c = RackDeltaCounter([RC], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    assert c.process(None, [], 11.0, _det([person, *two])) == []


def test_always_count_rack_counts_despite_person_present():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    ev = c.process(None, [], 11.0, _det([person, *two]))
    assert len(ev) == 2


def test_switch_credits_old_washer_and_rebaselines():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    one = [_plate(20)]
    c.process("Wife", [], 5.0, _det(one))
    c.process("Wife", [], 5.5, _det(one))
    ev = c.process("Wife", [], 6.0, _det(one))
    assert len(ev) == 1 and ev[0].person == "You"
    three = [_plate(20), _plate(40), _plate(60)]
    c.process(None, [], 10.0, _det(three))
    c.process(None, [], 10.5, _det(three))
    ev = c.process(None, [], 11.0, _det(three))
    assert len(ev) == 2 and all(e.person == "Wife" for e in ev)


def test_detect_called_only_during_bursts_and_throttled():
    calls = []

    def detect():
        calls.append(True)
        return []

    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.5)
    c.process(None, [], 0.0, detect)        # idle -> no call
    assert calls == []
    c.process("You", [], 1.0, detect)       # edge -> call 1
    c.process("You", [], 1.2, detect)       # throttled
    c.process("You", [], 1.6, detect)       # call 2
    c.process("You", [], 2.0, detect)       # finalize; throttled
    c.process("You", [], 3.0, detect)       # active, not collecting
    assert len(calls) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rackwash/tests/test_rack_delta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.rack_delta'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/rack_delta.py`**

```python
"""Outcome-based counting: at each session boundary, run a short burst of dish
counts over the rack regions and credit the net increase to the washer whose
session just closed. YOLO runs only during bursts (via the injected `detect`
callback). A requires_clear rack is sampled only on frames where no person/hand
overlaps it; a rack only counts when measured at BOTH boundaries."""

from __future__ import annotations

import statistics
from collections.abc import Callable

from rackwash.config import RackZone
from rackwash.domain import Dish, Hand, WashEvent
from rackwash.rack import count_dishware, rack_occluded


class RackDeltaCounter:
    def __init__(
        self,
        rack_zones: list[RackZone],
        dishware_labels: set[str],
        rack_window: float = 1.5,
        dish_interval: float = 0.5,
    ) -> None:
        self._zones = list(rack_zones)
        self._labels = set(dishware_labels)
        self._window = rack_window
        self._interval = dish_interval
        n = len(self._zones)
        self._washer: str | None = None
        self._baseline: list[int | None] = [None] * n
        self._prev_active: str | None = None
        self._collecting = False
        self._burst_end = 0.0
        self._last_run: float | None = None
        self._samples: list[list[int]] = [[] for _ in self._zones]
        self._closing_washer: str | None = None
        self._closing_baseline: list[int | None] = []
        self._new_washer: str | None = None

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    def process(
        self,
        active_washer: str | None,
        hands: list[Hand],
        now: float,
        detect: Callable[[], list[Dish]] | None,
    ) -> list[WashEvent]:
        if active_washer != self._prev_active:
            self._start_burst(active_washer, now)
        self._prev_active = active_washer

        events: list[WashEvent] = []
        if self._collecting:
            self._maybe_sample(hands, now, detect)
            if now >= self._burst_end:
                events = self._finalize(now)
        return events

    def _start_burst(self, active_washer: str | None, now: float) -> None:
        if not self._collecting:
            self._collecting = True
            self._closing_washer = self._washer
            self._closing_baseline = list(self._baseline)
            self._samples = [[] for _ in self._zones]
            self._last_run = None
        self._new_washer = active_washer  # coalesce: latest active wins
        self._burst_end = now + self._window

    def _maybe_sample(
        self, hands: list[Hand], now: float, detect: Callable[[], list[Dish]] | None
    ) -> None:
        if detect is None:
            return
        if self._last_run is not None and (now - self._last_run) < self._interval:
            return
        self._last_run = now
        try:
            dishes = detect()
        except Exception:  # noqa: BLE001 - a flaky detector must not crash the loop
            return
        persons = [d for d in dishes if d.label == "person"]
        for i, zone in enumerate(self._zones):
            if zone.requires_clear and rack_occluded(zone, persons, hands):
                continue
            self._samples[i].append(count_dishware(dishes, zone, self._labels))

    def _finalize(self, now: float) -> list[WashEvent]:
        end = [statistics.median_low(s) if s else None for s in self._samples]
        events: list[WashEvent] = []
        if self._closing_washer in ("You", "Wife"):
            delta = 0
            for base, fin in zip(self._closing_baseline, end):
                if base is not None and fin is not None:
                    delta += max(0, fin - base)
            events = [
                WashEvent(
                    person=self._closing_washer, timestamp=now,
                    confidence=1.0, source_id=-1,
                )
                for _ in range(delta)
            ]
        self._baseline = end
        self._washer = self._new_washer
        self._collecting = False
        return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rackwash/tests/test_rack_delta.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/rack_delta.py rackwash/tests/test_rack_delta.py
git commit -m "feat: RackDeltaCounter for outcome-based per-session counting

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `engine.py` + `annotate` + conftest

**Files:**
- Create: `rackwash/src/rackwash/engine.py`
- Create: `rackwash/tests/conftest.py`, `rackwash/tests/test_engine_e2e.py`, `rackwash/tests/test_annotate.py`

**Interfaces:**
- Consumes: everything above + the injected dish detector.
- Produces:
  - `Engine(camera, detector, config, store, state, clock=time.time, jpeg_encoder=encode_jpeg, annotator=annotate, dish_detector=None)`.
  - `process_frame(frame, now) -> list[WashEvent]`; `run()`; `stop()`; `_resolve_gesture(hands) -> str` (no location filter).
  - `annotate(frame, config, hands, active_washer, gesture, collecting=False) -> ndarray`.

- [ ] **Step 1: Write the fixtures and failing tests**

Create `rackwash/tests/conftest.py`:
```python
import numpy as np
import pytest

from rackwash.config import Config, RackZone, Thresholds


@pytest.fixture
def sample_config() -> Config:
    return Config(
        camera_index=0,
        rack_zones=[RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False)],
        thresholds=Thresholds(gesture_hold=1.5, rack_window=1.0, dish_interval=0.4),
    )


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((120, 200, 3), dtype=np.uint8)
```

Create `rackwash/tests/test_annotate.py`:
```python
import numpy as np

from rackwash.config import Config, RackZone
from rackwash.domain import Hand
from rackwash.engine import annotate


def _cfg() -> Config:
    return Config(
        camera_index=0,
        rack_zones=[RackZone(x1=0, y1=0, x2=50, y2=50, requires_clear=False)],
    )


def test_annotate_runs_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [Hand(id=1, bbox=(10, 10, 30, 30))], "You", "one")
    assert out.shape == frame.shape
    assert out.any()


def test_annotate_no_session_or_hands():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    assert annotate(frame, _cfg(), [], None, "other").shape == frame.shape


def test_annotate_collecting_marker_adds_pixels():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    base = annotate(frame, _cfg(), [], "You", "other", False)
    marked = annotate(frame, _cfg(), [], "You", "other", True)
    assert int(marked.sum()) > int(base.sum())
```

Create `rackwash/tests/test_engine_e2e.py`:
```python
import numpy as np

from rackwash.camera import FakeCamera
from rackwash.detector import FakeHandDetector
from rackwash.domain import Dish, Hand
from rackwash.engine import Engine
from rackwash.state import SharedState
from rackwash.store import CountStore

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


def _plate(cx, cy=50):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label="plate", confidence=0.9)


class _RackFake:
    def __init__(self):
        self.dishes = []

    def detect(self, frame):
        return list(self.dishes)


def _engine(cfg, hand_script, rack):
    return Engine(
        FakeCamera([]),
        FakeHandDetector(hand_script),
        cfg,
        CountStore(":memory:"),
        SharedState(),
        clock=lambda: 0.0,
        jpeg_encoder=lambda frame: b"jpeg",
        dish_detector=rack,
    )


def test_gesture_registers_anywhere(sample_config):
    engine = _engine(sample_config, [[]], _RackFake())
    assert engine._resolve_gesture([_hand(50, 50, {"index"})]) == "one"


def test_full_session_counts_rack_delta_for_you(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})
    none = []
    rack = _RackFake()
    script = [[one], [one], none, none, [one], [one], none, none]
    engine = _engine(sample_config, script, rack)

    engine.process_frame(blank_frame, 0.0)   # one (hold begins)
    engine.process_frame(blank_frame, 1.5)   # one held 1.5 -> You; start burst sample 0
    engine.process_frame(blank_frame, 1.9)   # burst sample 0
    engine.process_frame(blank_frame, 2.5)   # finalize start burst (baseline 0)

    rack.dishes = [_plate(20), _plate(40)]

    engine.process_frame(blank_frame, 5.0)   # one (hold begins)
    engine.process_frame(blank_frame, 6.5)   # one held 1.5 -> session ends; end burst sample 2
    engine.process_frame(blank_frame, 6.9)   # burst sample 2
    engine.process_frame(blank_frame, 7.5)   # finalize -> delta 2

    assert engine._store.totals(7.5)["all_time"] == {"You": 2, "Wife": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest rackwash/tests/test_engine_e2e.py rackwash/tests/test_annotate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.engine'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/engine.py`**

```python
"""Wires the vision pipeline and owns the capture loop. Dependency-injected so
the whole pipeline runs in tests with zero hardware. Counting is outcome-based:
a RackDeltaCounter counts dishware added to the racks between gesture-marked
session boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from rackwash.config import Config
from rackwash.domain import WashEvent
from rackwash.gesture import recognize_gesture
from rackwash.rack_delta import RackDeltaCounter
from rackwash.session import SessionController
from rackwash.state import SharedState
from rackwash.store import CountStore
from rackwash.tracker import IouTracker


def encode_jpeg(frame: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate(frame: np.ndarray, config: Config, hands, active_washer, gesture,
             collecting: bool = False) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    for rz in config.rack_zones:
        color = (0, 165, 255) if rz.requires_clear else (255, 0, 0)
        cv2.rectangle(out, (rz.x1, rz.y1), (rz.x2, rz.y2), color, 2)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    if active_washer:
        banner, color = f"Session: {active_washer}", (0, 255, 0)
    else:
        banner = "Session: none - show 1 (You) / 2 (Wife), show again to end"
        color = (0, 165, 255)
    cv2.putText(out, banner, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"gesture: {gesture}", (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if collecting:
        cv2.putText(out, "counting rack...", (8, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


class Engine:
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
        self._camera = camera
        self._detector = detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._dish_detector = dish_detector
        self._running = False

        t = config.thresholds
        self._hand_tracker = IouTracker(
            iou_threshold=t.iou_match, coast_seconds=t.track_coast
        )
        self._session = SessionController(hold_seconds=t.gesture_hold)
        self._rack = RackDeltaCounter(
            config.rack_zones, set(config.dish_classes),
            rack_window=t.rack_window, dish_interval=t.dish_interval,
        )

    def _resolve_gesture(self, hands) -> str:
        for hand in hands:
            g = recognize_gesture(hand)
            if g in ("one", "two"):
                return g
        return "other"

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame), now)
        gesture = self._resolve_gesture(hands)
        active = self._session.update(gesture, now)
        detect = (
            (lambda: self._dish_detector.detect(frame))
            if self._dish_detector is not None
            else None
        )
        events = self._rack.process(active, hands, now, detect)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(
            frame, self._config, hands, active, gesture, self._rack.is_collecting
        )
        self._state.publish(
            self._encode(annotated), self._store.totals(now), camera_online=True
        )
        return events

    def run(self) -> None:
        self._running = True
        backoff = 0.5
        while self._running:
            frame = self._camera.read()
            if frame is None:
                if isinstance(getattr(self._camera, "_frames", None), list):
                    break
                self._state.publish(None, self._store.totals(self._clock()), False)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            self.process_frame(frame, self._clock())

    def stop(self) -> None:
        self._running = False
```

- [ ] **Step 4: Run tests, then the whole rackwash suite + lint**

Run:
```bash
.venv/bin/python -m pytest rackwash/tests/test_engine_e2e.py rackwash/tests/test_annotate.py -v
.venv/bin/python -m pytest rackwash/tests -q
.venv/bin/ruff check rackwash
```
Expected: the e2e + annotate tests pass; the whole rackwash suite passes; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/engine.py rackwash/tests/conftest.py rackwash/tests/test_engine_e2e.py rackwash/tests/test_annotate.py
git commit -m "feat: rackwash engine wiring rack-delta counting end to end

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `calibrate.py` — draw rack zones with `requires_clear`

**Files:**
- Create: `rackwash/src/rackwash/calibrate.py`
- Test: `rackwash/tests/test_calibrate.py`

**Interfaces:**
- Consumes: `RackZone`/`Config`/`Thresholds` (T4).
- Produces:
  - `rack_zone_from_drag(start, end, requires_clear) -> RackZone`.
  - `run_calibration(config_path="config.yaml", camera_index=0) -> None` (interactive, `# pragma: no cover`).

- [ ] **Step 1: Write the failing test**

Create `rackwash/tests/test_calibrate.py`:
```python
from rackwash.calibrate import rack_zone_from_drag


def test_rack_zone_from_drag_normalizes_and_sets_flag():
    rz = rack_zone_from_drag((120, 90), (20, 10), requires_clear=True)
    assert (rz.x1, rz.y1, rz.x2, rz.y2) == (20, 10, 120, 90)
    assert rz.requires_clear is True

    rz2 = rack_zone_from_drag((0, 0), (50, 40), requires_clear=False)
    assert rz2.requires_clear is False
    assert rz2.contains((25, 20)) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rackwash/tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.calibrate'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/calibrate.py`**

```python
"""One-time (re-runnable) calibration: drag a rectangle around each drying rack
and mark whether your body blocks it while washing. Persists only the rack
rectangles — never raw images. Identity is by gesture."""

from __future__ import annotations

from pathlib import Path

from rackwash.config import Config, RackZone, Thresholds


def rack_zone_from_drag(
    start: tuple[int, int], end: tuple[int, int], requires_clear: bool
) -> RackZone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return RackZone(x1=x1, y1=y1, x2=x2, y2=y2, requires_clear=requires_clear)


def run_calibration(
    config_path: str | Path = "config.yaml", camera_index: int = 0
) -> None:  # pragma: no cover - interactive, exercised manually
    import cv2  # noqa: PLC0415

    from rackwash.camera import Camera

    cam = Camera(camera_index)
    print(
        "Calibration: for each drying rack, drag a rectangle and press ENTER.\n"
        "Then press 'b' if your body blocks that rack while washing, else 'a'.\n"
        "Press ESC (no drag) when you have added all racks."
    )
    rack_zones: list[RackZone] = []
    while True:
        frame = None
        while frame is None:
            frame = cam.read()
        roi = cv2.selectROI("drag a rack zone (ESC to finish)", frame, showCrosshair=True)
        x, y, bw, bh = (int(v) for v in roi)
        if bw == 0 or bh == 0:
            break
        key = -1
        while key not in (ord("a"), ord("b")):
            key = cv2.waitKey(0) & 0xFF
        rack_zones.append(rack_zone_from_drag((x, y), (x + bw, y + bh), key == ord("b")))
    cv2.destroyAllWindows()
    cam.release()

    if not rack_zones:
        print("No rack zones drawn; nothing saved.")
        return
    Config(
        camera_index=camera_index, rack_zones=rack_zones, thresholds=Thresholds()
    ).save(config_path)
    print(f"Saved {len(rack_zones)} rack zone(s) to {config_path}.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest rackwash/tests/test_calibrate.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/calibrate.py rackwash/tests/test_calibrate.py
git commit -m "feat: rack-zone calibration with requires_clear

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: `cli.py` — `rackwash calibrate` / `run`

**Files:**
- Create: `rackwash/src/rackwash/cli.py`
- Test: `rackwash/tests/test_cli.py`

**Interfaces:**
- Consumes: `Config` (T4), `calibrate.run_calibration` (T8), and (lazily) `Engine`/detectors/store/state/dashboard.
- Produces: `build_parser()`; `main(argv=None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `rackwash/tests/test_cli.py`:
```python
import yaml

from rackwash.cli import build_parser, main


def test_parser_has_calibrate_and_run():
    p = build_parser()
    assert p.parse_args(["calibrate", "--camera", "2"]).camera == 2
    assert p.parse_args(["run", "--port", "9000"]).port == 9000


def test_run_missing_config_exits_2_with_calibrate_hint(tmp_path, capsys):
    code = main(["run", "--config", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "calibrate" in capsys.readouterr().err


def test_run_empty_rack_zones_exits_2_with_rack_hint(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"camera_index": 0, "rack_zones": []}))
    code = main(["run", "--config", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "calibrate" in err and "rack" in err


def test_no_command_returns_1(capsys):
    assert main([]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest rackwash/tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rackwash.cli'`.

- [ ] **Step 3: Write `rackwash/src/rackwash/cli.py`**

```python
"""Command-line entry points: `rackwash calibrate` and `rackwash run`."""

from __future__ import annotations

import argparse
import sys
import threading

from rackwash.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rackwash")
    sub = parser.add_subparsers(dest="command")

    cal = sub.add_parser("calibrate", help="draw your drying-rack regions")
    cal.add_argument("--config", default="config.yaml")
    cal.add_argument("--camera", type=int, default=0)

    run = sub.add_parser("run", help="start the counter + dashboard")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--db", default="rackwash.db")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        from rackwash.calibrate import run_calibration  # noqa: PLC0415

        run_calibration(args.config, args.camera)
        return 0

    if args.command == "run":
        try:
            config = Config.load(args.config)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not config.rack_zones:
            print(
                "No rack zones configured. Run `rackwash calibrate` first to draw "
                "your rack region(s).",
                file=sys.stderr,
            )
            return 2
        _serve(config, args.db, args.host, args.port)
        return 0

    parser.print_help(sys.stderr)
    return 1


def _serve(config: Config, db: str, host: str, port: int) -> None:  # pragma: no cover
    from rackwash.camera import Camera  # noqa: PLC0415
    from rackwash.dashboard import run_server  # noqa: PLC0415
    from rackwash.detector import MediaPipeHandDetector  # noqa: PLC0415
    from rackwash.dish_detector import YoloWorldDishDetector  # noqa: PLC0415
    from rackwash.engine import Engine  # noqa: PLC0415
    from rackwash.state import SharedState  # noqa: PLC0415
    from rackwash.store import CountStore  # noqa: PLC0415

    state = SharedState()
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        config,
        CountStore(db),
        state,
        dish_detector=YoloWorldDishDetector(
            [*config.dish_classes, "person"], config.dish_conf, config.yolo_model
        ),
    )
    threading.Thread(target=engine.run, daemon=True).start()
    print(f"Dashboard at http://{host}:{port}")
    run_server(state, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests + verify the console script**

Run:
```bash
.venv/bin/python -m pytest rackwash/tests/test_cli.py -v
.venv/bin/python -c "import rackwash.cli"
.venv/bin/rackwash --help
```
Expected: tests pass; import succeeds (no `ultralytics` pulled in); help lists `calibrate` and `run`.

- [ ] **Step 5: Commit**

```bash
git add rackwash/src/rackwash/cli.py rackwash/tests/test_cli.py
git commit -m "feat: rackwash CLI with calibrate and run (rack-zone guard)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: README and final sign-off

**Files:**
- Create: `rackwash/README.md`

**Interfaces:**
- Consumes: everything.
- Produces: operator docs; a green rackwash suite and clean lint.

- [ ] **Step 1: Write `rackwash/README.md`**

```markdown
# Rackwash

A local computer-vision scoreboard that tallies how many dishes **You** vs
**Wife** wash — by counting the **net increase of dishware in your drying racks**
over a washing session, not by watching hands. Outcome-based: a count only
happens when a clean dish actually appears in a rack.

Standalone package — it does not depend on `dishcounter`.

## Requirements

- macOS (or Linux) with a webcam.
- Python 3.12 (MediaPipe has no 3.14 support): `brew install python@3.12`.

## Setup

Installs into the shared project venv (reuses the heavy MediaPipe/YOLO wheels):

\`\`\`bash
.venv/bin/python -m pip install -e ./rackwash
\`\`\`

## Calibrate

\`\`\`bash
.venv/bin/rackwash calibrate
\`\`\`

For each drying rack, drag a rectangle and press **Enter**, then press **`b`** if
your body blocks that rack while washing, or **`a`** if it stays visible. Press
**Esc** (without dragging) when you've added all your racks. Saves `config.yaml`
(rack rectangles only — no images).

## Run

\`\`\`bash
.venv/bin/rackwash run        # open http://127.0.0.1:8000
\`\`\`

Start a session by holding up **1 finger** (You) or **2** (Wife) for ~1.5s,
anywhere in view; show the same number again to end it. The app counts the
dishware added to your racks during the session and credits it to that washer.
On first run, MediaPipe and YOLO-World download their weights (~hundreds of MB).
YOLO runs only in short bursts at the start and end of each session.

## How it works

\`\`\`
Camera -> MediaPipe(hands) -> IouTracker -> recognize_gesture -> SessionController
       -> RackDeltaCounter (YOLO bursts at boundaries) -> CountStore
\`\`\`

At each session boundary the `RackDeltaCounter` runs a short YOLO burst over the
rack regions, takes a robust per-rack dishware count (median over clean frames,
skipping person/hand-occluded frames for racks you marked as body-blocked), and
credits `max(0, end - start)` per rack to the session's washer. Counts persist in
`rackwash.db`.

## Develop

\`\`\`bash
.venv/bin/python -m pytest rackwash/tests -q
.venv/bin/ruff check rackwash
\`\`\`

## Privacy

Only the rack rectangles (`config.yaml`) and event rows (`rackwash.db`) are
stored. No images are ever persisted.
```

- [ ] **Step 2: Final rackwash suite + lint sign-off**

Run:
```bash
.venv/bin/python -m pytest rackwash/tests -q
.venv/bin/ruff check rackwash
```
Expected: all rackwash tests pass, pristine (no warnings); ruff `All checks passed!`. If either is not clean, STOP and fix before committing.

- [ ] **Step 3: Commit**

```bash
git add rackwash/README.md
git commit -m "docs: rackwash operator README

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

- Standalone package, no dishcounter import/modify → Tasks 1–3 (own pyproject; copies re-namespaced), all `rackwash.` imports. ✓
- Outcome-based per-session counting (boundary bursts, median, per-rack clamped delta, both-boundaries rule) → Task 6 (`RackDeltaCounter`), wired Task 7. ✓
- Gestures anywhere, 1.5s hold → Task 4 (`gesture_hold=1.5`), Task 7 (`_resolve_gesture` no filter), tested `test_gesture_registers_anywhere`. ✓
- Per-rack `requires_clear` + person/hand occlusion → Task 4/5/6, tested. ✓
- Dishware-only classes; YOLO `+ ["person"]` → Task 4 (`dish_classes`), Task 9 (`_serve`). ✓
- YOLO only during bursts via `detect` callback → Task 6, tested `test_detect_called_only_during_bursts_and_throttled`. ✓
- `source_id=-1`, labels You/Wife, undercount clamp → Task 6. ✓
- `run` refuses empty `rack_zones` → Task 9, tested. ✓
- Calibration draws rack zones + `requires_clear` → Task 8. ✓
- Detector exception → skip sample → Task 6 `_maybe_sample`. ✓

**Placeholder scan:** No TBD/TODO. New modules have full code; copied modules name exact source files + the one mechanical edit (`from dishcounter.` → `from rackwash.`); interactive `run_calibration` is `# pragma: no cover` with its pure helper tested.

**Type consistency:** `RackDeltaCounter(rack_zones, dishware_labels, rack_window, dish_interval)` + `process(active, hands, now, detect)` + `is_collecting` match between Task 6 (def) and Task 7 (engine). `count_dishware(dishes, zone, dishware_labels)` / `rack_occluded(zone, persons, hands)` match Task 5 (def) and Task 6 (use). `RackZone(x1,y1,x2,y2,requires_clear)` + `Zone.overlaps(bbox)` consistent Tasks 4–8. `Config(camera_index, rack_zones, thresholds, dish_classes, dish_conf, yolo_model)` consistent Tasks 4/7/8/9. `annotate(..., collecting=False)` matches Task 7 def + `process_frame` call. `Engine(..., dish_detector=None)` matches Task 7 def, e2e tests, and Task 9 `_serve`.
```
