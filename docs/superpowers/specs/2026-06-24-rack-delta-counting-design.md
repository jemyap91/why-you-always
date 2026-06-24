# Rack-delta counting — design

**Date:** 2026-06-24
**Status:** Approved (pre-implementation)

## Problem

Every counting mechanism so far measured **hand activity** and inferred a wash
from it: the sink→drying zone crossing, then hand dwell-and-leave, then the
dish-presence gate. All overcount, because hand activity is not washing — a hand
rinsing, repositioning, or grabbing a sponge dwells in the sink and counts,
especially when a dirty dish is already sitting there. The mechanism, not the
tuning, is the problem.

## Goal

Count the **outcome**, not the activity: how many clean dishes actually ended up
in the drying racks during a washing session. A session is a period with one
active washer (set by gesture). Measure the dishware in the racks at the start of
the session and again at the end; the increase is the number washed, credited to
that washer.

> One session's count = (dishware in the racks at session end) − (dishware in the
> racks at session start), clamped to ≥ 0, attributed to the session's washer.

This is almost impossible to inflate: a count only happens when a physical dish
appears in a rack. YOLO is used to **count dishware in a static region**, never
to track or detect washing — sidestepping the per-dish-tracking unreliability
that sank the earlier dish-detection attempt.

## Non-goals

- Counting from hand motion in any form (the entire `WashCycleEngine` /
  dish-presence-gate / sink-zone counting is removed).
- Per-dish event tracking across frames.
- Counting cutlery (forks/knives/spoons) — dishware only.
- Handling a session that both removes dry dishes and adds washed ones (the user
  rarely does this; a simple end−start delta is accurate, and removals only ever
  cause a safe undercount).

## Approach (counting per session via boundary bursts)

Reuse the working pieces — MediaPipe hands, the IoU tracker, gesture recognition,
and the gesture-driven session — to mark **session boundaries**. Replace all
hand-activity counting with a `RackDeltaCounter` that, at each boundary, runs a
short **burst** of YOLO detections over the rack regions and records a robust
dishware count.

### Pipeline

```
Camera ─ MediaPipe(hands) ─ IouTracker ─ recognize_gesture ─ SessionController(hold=1.5) ─┐
                                                                                          │ active washer
                                           DishDetector (YOLO: dishware + person, burst) ─┤
                                                                                          ▼
                                                                             RackDeltaCounter ─► CountStore
```

### Sessions and boundaries

- A **session** = a contiguous period with one active washer (`"You"`/`"Wife"`).
- **Boundaries**: start (none→washer), end (washer→none), switch (washer→other).
- Gestures register **anywhere in frame**: any visible hand showing 1 finger
  (You) or 2 (Wife), held continuously for `gesture_hold` (**1.5s**), toggles
  that person. There is **no sink zone** — the washing hand is normally hidden
  behind the user's body (back to camera) so it cannot register; the 1.5s hold
  guards the rare visible-hand case. (`SessionController` is unchanged except the
  default hold; `recognize_gesture` is unchanged; the engine's gesture resolver
  no longer filters by location.)

### The burst

At every boundary the counter runs a burst for up to `rack_window` (**1.5s**):

- Each frame (throttled to at most one YOLO run per `dish_interval`, default
  0.5s), run YOLO over the frame with classes = `dish_classes` + `"person"`.
- For **each rack zone** independently, that frame contributes a sample
  `count_dishware(dishes, rack_zone)` (dishware detections whose centroid is in
  the zone) **iff the rack is countable this frame**:
  - a rack with `requires_clear = False` is **always** countable;
  - a rack with `requires_clear = True` is countable only if **no `person`
    detection and no tracked hand overlaps that rack's rectangle** this frame.
- At burst end, each rack's robust count is the **median** of its samples (or
  `None` if it gathered no countable samples).

Per-rack occlusion (not whole-frame) matches reality: the main rack is never
blocked by the user's body (`requires_clear = False`, counted every frame), while
the over-sink rack is blocked while washing (`requires_clear = True`, counted
only once the user steps clear). YOLO runs **only during bursts** — idle while
washing and while no session is active.

### Closing a session

The counter keeps the current session's washer and a **per-rack baseline** (each
rack's robust count captured at the session's start boundary, or `None` if that
rack had no clean sample there). When a boundary's burst finalizes:

1. Compute each rack's robust end count `end[r]` (`None` if no countable samples).
2. If a session was in progress (the closing washer is not `None`): for each rack,
   `delta_r = max(0, end[r] − baseline[r])` **only when both `baseline[r]` and
   `end[r]` are measured**, else `delta_r = 0`. A rack with no clean reading at
   *either* boundary is not counted this session — a safe undercount that can
   never over-credit (e.g. an occluded baseline must not make a full rack look
   freshly washed). `delta = Σ_r delta_r` (per-rack clamping so one rack's
   miscount can't cancel another's real gain). Emit `delta` identical
   `WashEvent`s for the closing washer.
3. Set `baseline = end` (per rack, `None`-preserving) and the current washer to
   the new active washer. A **switch** thus reuses the switch-point counts as
   both the old session's end and the new session's start.

So a `requires_clear` rack counts only when it is clear at **both** the start and
end boundaries; the always-on main rack (`requires_clear=False`) is measured at
every boundary and counts normally.

Rapid re-toggles inside a burst window are coalesced: the burst finalizes once,
and the latest active washer becomes the next session's washer.

## Components

### Shared types (`domain.py`)
- `Dish` (existing) gains nothing; YOLO now also returns `Dish(label="person", …)`
  detections, distinguished by `label`.
- `WashEvent` (existing): rack-derived events use `source_id = -1` (sentinel for
  "session/rack-derived, not hand-derived").

### `config.py`
- **Remove** `sink_zone`. **Add** `rack_zones: list[RackZone]`.
- New `RackZone(BaseModel)`: `x1, y1, x2, y2, requires_clear: bool = True`, with
  `contains(point)` (centroid test, normalized corners) and `overlaps(bbox)`
  (rectangle intersection, normalized corners).
- `dish_classes` default → `["plate", "bowl", "cup", "glass", "mug"]` (no cutlery).
- `Thresholds`: keep `gesture_hold` (default → **1.5**), `iou_match`,
  `track_coast`; add `rack_window: float = 1.5`; keep `dish_interval: float = 0.5`.
  **Remove** `min_wash`, `cooldown`, `dish_min_hits`. `dish_conf` stays on `Config`.
- An existing `config.yaml` lacking `rack_zones` fails the "are there rack zones?"
  start check (below) → user re-calibrates.

### `rack.py` (new) — pure counting helpers
- `count_dishware(dishes: list[Dish], zone: RackZone, dishware_labels: set[str]) -> int`
  — dishware detections (label in `dishware_labels`) whose centroid is in `zone`.
- `rack_occluded(zone: RackZone, persons: list[Dish], hands: list[Hand]) -> bool`
  — any person `Dish` bbox or any hand bbox overlaps `zone`.

### `rack_delta.py` (new) — `RackDeltaCounter` (the heart, replaces `washcycle.py`)
- `RackDeltaCounter(rack_zones, dishware_labels, rack_window=1.5, dish_interval=0.5)`.
- `process(active_washer: str | None, hands: list[Hand], now: float, detect: Callable[[], list[Dish]] | None) -> list[WashEvent]`.
  - `detect` is a zero-arg callable the engine supplies; the counter calls it
    **only** when it needs a burst sample (collecting + throttle elapsed), so
    YOLO runs solely during bursts. A `detect()` exception is caught → that
    sample is skipped.
  - Holds the state machine, per-rack baselines, current washer, and the active
    burst's per-rack samples; emits events on burst finalize as above.
- `is_collecting` (property) — whether a burst is in progress (for the overlay).

### `engine.py`
- Remove `WashCycleEngine`, the sink-zone gesture filter, and `_maybe_detect_dish`.
- `_resolve_gesture(hands)` returns the first hand showing `one`/`two` (no location
  filter).
- `process_frame`: detect+track hands → resolve gesture → `active = session.update`
  → `events = self._rack.process(active, hands, now, detect=lambda: self._dish_detector.detect(frame))`
  → record events → annotate → publish.
- `annotate(frame, config, hands, active_washer, gesture, collecting)` draws each
  rack rectangle (tinted by `requires_clear`), the session banner, the `gesture:`
  label, and a "counting rack…" marker while `collecting`. No sink/dish-in-sink
  drawing.
- The dish detector is constructed with classes `dish_classes + ["person"]`.

### `calibrate.py`
- Draw **rack zones only** (no sink). For each rack: drag the rectangle, then a
  keypress sets `requires_clear` (e.g. `b` = "my body blocks this rack" → True,
  `a` = "always visible" → False; default True). Repeat until the user finishes.
  Saved as `rack_zones`. Pure helpers (`zone_from_drag`, profile-free) stay
  testable; the interactive loop stays `# pragma: no cover`.

### `cli.py`
- `_serve`: refuse with a clear "no rack zones — run `dishcounter calibrate`
  first" message if `config.rack_zones` is empty. Build
  `YoloWorldDishDetector(config.dish_classes + ["person"], config.dish_conf, config.yolo_model)`
  and pass it to `Engine`.

## Attribution & storage

Identity is the session's washer at the closing boundary. Each washed dish is one
counted `WashEvent` (`person`, `confidence=1.0`, `source_id=-1`). `CountStore` and
the event-sourced today/all-time totals are unchanged.

## Error handling / edge cases

- **YOLO load failure** surfaces at `run` startup, never mid-loop, never in tests.
- **Detector exception on a burst frame** → that sample is skipped; the engine
  continues.
- **Empty `rack_zones`** → `run` refuses up front.
- **A `requires_clear` rack blocked at a boundary** (no clear frame in that
  burst) → no clean reading → that rack contributes 0 to any session touching
  that boundary (safe; never over-credits). The user can re-gesture.
- **Dishes removed during a session** (end < start) → per-rack `max(0, …)` →
  safe undercount, never negative.
- **Reflections / transparent glassware** inflating a rack count → mitigated by
  drawing ROIs tightly to the shelf (below the reflective backsplash) and by the
  per-rack median over the burst. Residual risk; this is the design's main
  accuracy dependency.
- **Rapid re-toggle within a burst** → coalesced to one finalize.
- Per-session state is bounded (one burst's samples + per-rack baselines).

## Testing

All detection behind dependency-injected fakes / a `detect` callback; **no
ultralytics import in tests**.

- **`rack` helpers:** `count_dishware` counts only in-zone dishware of the right
  labels; `rack_occluded` true iff a person/hand bbox overlaps the zone;
  `RackZone.contains`/`overlaps` correctness incl. normalized corners.
- **`RackDeltaCounter`:**
  - start→end with N dishware added to a rack → N events for the washer;
  - nothing added → 0 events;
  - dishes removed (end<start) → 0 (clamped);
  - a flicker frame in a burst is discarded by the median;
  - `requires_clear=True` rack occluded by a person every end-frame → that rack
    contributes 0; `requires_clear=False` rack counted despite a person present;
  - a mid-session **switch** credits the old washer to the switch point and
    re-baselines the new washer;
  - YOLO (`detect`) is called **only during bursts** and respects `dish_interval`
    (spy callback counts calls).
- **engine e2e:** scripted gestures + a scripted `detect` returning rack dishes →
  correct delta events; gestures register with no sink zone; the overlay's
  `collecting` flag is set during bursts.
- **config / calibrate:** `rack_zones` (with `requires_clear`) round-trips;
  `dish_classes` excludes cutlery; `run` refuses on empty `rack_zones`; calibrate
  zone helpers tested.

## Out of scope / future

- Sessions that remove and add dishes simultaneously (net-change tracking).
- Cropping each rack ROI before YOLO for speed/accuracy.
- A live per-rack count readout on the overlay.
- Counting cutlery.
