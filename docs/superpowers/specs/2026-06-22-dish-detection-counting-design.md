# Dish-detection counting — design

**Date:** 2026-06-22
**Status:** Approved (pre-implementation)

## Problem

There is no physical space for a drying rack after the sink, so the original
two-zone counting rule (hand crosses `sink_zone` → `drying_zone`) no longer
matches reality. The intended workflow is now:

> A dish is washed at the sink, then carried out of the camera's view to be put
> away. "Dish leaves view" == one dish washed.

We also want the count to reflect **actual dishes**, not a hand proxy — so an
empty-handed exit must not count, and a real plate leaving must.

## Goal

Count each washed dish when the **dish object** leaves the frame after being
washed at the sink, and attribute it to **You** or **Wife** via the nearest
hand's skin tone. Today's totals and all-time totals (event-sourced in SQLite)
are unchanged downstream.

## Non-goals

- Counting how many dishes are stacked/carried in a single grab (each detected
  dish box is one dish; an undetected second plate behind the first is missed).
- Training a custom model. We use an open-vocabulary off-the-shelf model.
- Changing the storage layer, dashboard, or identity (skin-tone) algorithm.

## Approach

Two detectors run per frame and are **fused**:

- **YOLO-World (Ultralytics)** detects dishes — it counts. Open-vocabulary so it
  detects `plate` (which COCO-pretrained YOLO cannot) plus bowls/cups/cutlery
  from text-prompt class names, with no training.
- **MediaPipe hand landmarker** (existing) supplies identity — the nearest
  hand's locked skin-tone classification says You vs Wife. A dish has no skin
  tone, so both detectors are required.

The original hand-exit counting idea is dropped: **dishes drive the count;
hands only attribute identity.**

### Pipeline

```
MediaPipe(hands) ─┐
                  ├─► HandTracker ──┐
YOLO-World(dishes)┘                 ├─► FusionEngine ─► WashEvent ─► CountStore
                  └─► DishTracker ──┘
```

### Trigger rule (FusionEngine)

Per tracked **dish** id, keep: `last_sink_time`, `last_seen`,
`locked_person`, `locked_confidence`.

Each `process(hands, dishes, now)`:

1. For each present dish: set `last_seen = now`. If the dish bbox overlaps
   `sink_zone` (centroid-in-zone, matching `Zone.contains`), stamp
   `last_sink_time = now` and, while identity is still `uncertain`, lock it to
   the **nearest hand's** identity — where "nearest" is the hand whose centroid
   is closest to the dish centroid, and that hand has a confident (non-uncertain)
   skin-tone classification. (Hand identity is classified exactly as today.)
2. Scan tracked dish ids **not** present this frame:
   - Gone for `> exit_grace` **and** had a sink visit → fire one `WashEvent`
     (`person = locked_person`, `confidence = locked_confidence`,
     `timestamp = now`); then drop the dish state.
   - Cooldown applies only to known persons (`You`/`Wife`): skip the fire if
     that person fired within `cooldown` seconds. `uncertain` exits are still
     emitted (the store records but does not count them — preserves the audit
     trail).
   - Gone for `> exit_grace` with **no** sink visit → drop the state silently
     (prevents unbounded per-id state growth; not a wash).

`exit_grace` must exceed normal detection flicker so a dish that briefly drops
out while being scrubbed does not falsely "exit." Default **1.5s**.

## Components

### New / changed domain types (`domain.py`)
- Add `Dish` dataclass: `id: int | None`, `bbox: (x1,y1,x2,y2)`, `label: str`,
  `confidence: float`, with a `centroid` property (mirrors `Hand`).

### `DishDetector` (new module, e.g. `dish_detector.py`)
- Wraps Ultralytics YOLO-World. `detect(frame) -> list[Dish]`.
- Loads the model once; sets open-vocabulary classes from config
  (`dish_classes`); filters detections below `dish_conf`.
- Mirrors `MediaPipeHandDetector` so it is dependency-injected and never loaded
  in tests (tests inject a fake returning canned `Dish` lists).

### Generic tracker (`tracker.py`)
- Generalize the existing IoU `HandTracker` (already box-based) into a small
  tracker that assigns stable ids to any object exposing `.bbox` and accepting
  `.id`. Run one instance for hands, one for dishes.
- Keep the public behavior identical for hands (existing tests must still pass).

### `FusionEngine` (replaces `zones.py` `ZoneEventEngine`)
- New module (e.g. `fusion.py`). Constructor:
  `FusionEngine(sink_zone, identity, exit_grace=1.5, cooldown=3.0)`.
- `process(hands, dishes, now) -> list[WashEvent]` implements the trigger rule
  above. Holds per-dish state and a `_last_person_fire: dict[str, float]`.
- `zones.py` is removed.

### `config.py`
- Remove the `drying_zone` field.
- `Thresholds`: keep `identity_distance`, `iou_match`, `cooldown`; add
  `exit_grace: float = 1.5`. (Drop `presence_window`, which was sink-visit
  freshness for the old zone rule.)
- Add dish settings: `dish_classes: list[str]` (default
  `["plate","bowl","cup","glass","mug","fork","knife","spoon"]`),
  `dish_conf: float = 0.4`, and `yolo_model: str` (YOLO-World weights name/path,
  e.g. `"yolov8s-worldv2.pt"`).

### `engine.py`
- Construct both detectors, both trackers, and the `FusionEngine`.
- `process_frame` runs hand + dish detection, updates both trackers, calls
  `FusionEngine.process(hands, dishes, now)`, records events, publishes state.
  (No loop change: `process_frame` already runs every frame, so absent-dish
  exits are detected as `now` advances.)
- `annotate()` draws the sink rectangle and dish boxes with labels; the green
  drying-zone box is removed.

### `calibrate.py`
- Calibrate only the `sink` rectangle; drop `drying` from the drag loop and from
  the saved `Config`.

### Dependencies
- Add `ultralytics` to `pyproject.toml` and `requirements.txt`. Model weights
  download on first run (document this in the README run section).

## Data flow / attribution detail

Identity is locked onto the **dish**, not the hand, at wash time. This means the
hand that carries the dish out can differ from the hand that was nearest during
washing — we attribute to whoever was washing (nearest hand at sink), which
matches intent. If no confident hand was ever near the dish at the sink, the
dish exits as `uncertain` (recorded, not counted).

## Error handling

- YOLO model load failure surfaces at startup (`run` only), not mid-loop; tests
  never load it.
- Per-dish state is bounded: dropped on both fire and no-sink exit.
- No new storage failure modes; `WashEvent` shape and `CountStore` are unchanged.

## Performance

YOLO-World runs per frame alongside MediaPipe on M1 Pro (CPU/MPS), reducing
effective FPS to single digits. Acceptable for a kitchen-counter scoreboard.
Frame rate is not a correctness concern because counting is event-based on dish
disappearance, debounced by `exit_grace`.

## Testing

All detection is behind dependency-injected fakes (same pattern as `FakeCamera`),
so no model loads in tests.

- **FusionEngine**: sink-visited dish disappears > grace → one event for the
  nearest hand's person; flicker under grace then reappears → no event;
  dish never in sink then exits → no event; two same-person dish exits within
  cooldown → collapse to one count; dish with no confident nearby hand →
  `uncertain` (recorded, not counted); dish reappearing with a new id after a
  count → counts again.
- **Generic tracker**: existing hand-tracking tests still pass; dishes get
  stable ids; ids drop when a dish disappears.
- **DishDetector**: thin wrapper test with a stubbed model output mapped to
  `Dish` objects and confidence filtering (no real weights).
- **Config / calibrate / e2e**: fixtures drop `drying_zone`, add dish settings;
  end-to-end test drives a scripted hand+dish sequence through `Engine` with
  fakes and asserts one counted wash.

## Out of scope / future

- Counting stacked dishes carried together.
- Custom-trained dish model for higher precision.
- Distinguishing dish types in the scoreboard (we only count dishes).
