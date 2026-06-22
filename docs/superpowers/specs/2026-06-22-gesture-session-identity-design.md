# Gesture-controlled session identity — design

**Date:** 2026-06-22
**Status:** Approved (pre-implementation)

## Problem

Attributing a wash to You vs Wife by hand **skin tone** does not work: even with
careful calibration the two people's chroma profiles land ~2.6 apart (well under
the classifier's 5.0 ambiguity margin), so every wash is recorded as `uncertain`
and nothing is counted. A ring/jewelry probe (Phase 0) also failed to separate
reliably on-camera.

The couple **takes turns** — one person washes a whole session. So per-frame
automatic identity is unnecessary; a deliberate, reliable, hands-free signal at
the start and end of a session is enough.

## Goal

Identity comes from a **hand gesture**:

- Show **1 finger** → start a session as **You**.
- Show **2 fingers** → start a session as **Wife**.
- Show a **fist** → end the session.

Every dish counted while a session is active is attributed to that session's
person. Before any session starts (and after a fist), no session is active and
nothing is counted (safe default). This replaces the skin-tone identity path
entirely.

## Non-goals

- Distinguishing two people washing simultaneously (they take turns).
- Per-dish identity. Identity is per-session.
- Skin-tone or jewelry classification (removed).
- Recognizing gestures beyond `one` / `two` / `fist`.

## Approach

A gesture recognizer reads finger state from the existing MediaPipe hand
landmarks. A session controller turns **debounced** gestures into an
`active_washer` state. The `FusionEngine` attributes each sink-visited dish to
the active washer at the time it is washed (or `uncertain` if no session is
active). Dish detection, tracking, exit-grace, and cooldown are unchanged.

The skin-tone subsystem (`IdentityClassifier`, skin-profile calibration, the
`region_pixels` skin sample, and the ring probe) is removed, since it is fully
superseded.

### Pipeline

```
HandDetector ─ IouTracker ─ GestureRecognizer ─ SessionController ─┐
                                                                   ├─ FusionEngine ─ CountStore
DishDetector ─ IouTracker ─────────────────────────────────────────┘
```

## Components

### `gesture.py` (new, pure)
Decide a gesture from a hand's 21 normalized landmarks. No calibration, no
frame access — pure geometry, fully unit-testable.

- `Gesture` values: `"one"`, `"two"`, `"fist"`, `"other"`.
- A finger (index/middle/ring/pinky) is **extended** when its tip is farther
  from the wrist (landmark 0) than its PIP joint:
  `dist(tip, wrist) > dist(pip, wrist)`. Orientation-tolerant.
  - index: tip 8, pip 6; middle: tip 12, pip 10; ring: tip 16, pip 14;
    pinky: tip 20, pip 18.
- Thumb is ignored for counting (it is unreliable and not needed for 1/2/fist).
- Classification from the set of extended fingers:
  - `fist` — no fingers extended.
  - `one` — only the index extended.
  - `two` — only index and middle extended.
  - `other` — anything else (ignored downstream).
- `recognize_gesture(hand) -> str`. Returns `"other"` if landmarks are missing
  (`len(hand.landmarks) < 21`).

### `SessionController` (new)
Owns the active washer and debounces gestures so a fleeting pose does not flip
the session.

- State: `active: str | None` (start `None`); `_candidate: str | None`;
  `_candidate_since: float`.
- `update(gesture: str, now: float) -> str | None`:
  - Tracks how long the current candidate gesture has been continuously held.
    Resets the timer when the recognized gesture changes.
  - When a meaningful gesture (`one`/`two`/`fist`) has been held for
    `>= hold_seconds`, it **acts once** (idempotent until the gesture changes):
    `one → active="You"`, `two → active="Wife"`, `fist → active=None`.
  - Returns the new `active` value (for logging/overlay); no-op for `other`.
- `hold_seconds` default 1.0 (from config `gesture_hold`).
- If multiple hands are present, the controller is driven by the most
  confident / first recognized meaningful gesture for the frame (the engine
  passes a single resolved gesture — see engine).

### `FusionEngine` (changed)
- Constructor drops `IdentityClassifier`; gains nothing else (session state
  lives in the engine via `SessionController`, passed the active washer).
- `process(hands, dishes, now, active_washer)` — new `active_washer` parameter
  (`"You"`/`"Wife"`/`None`). When a dish's centroid is in the sink zone, lock
  its identity to `active_washer` if one is set and the dish is still
  `uncertain`; confidence is recorded as `1.0` for an active session (a
  deliberate signal, not a probabilistic match).
- Exit detection, `exit_grace`, per-person `cooldown`, and bounded state are
  unchanged. `uncertain` dishes (washed with no active session) are still
  recorded with `counted=0`.

### `engine.py` (changed)
- Owns a `SessionController` and calls `gesture.recognize_gesture(...)`.
- `process_frame`: detect hands → recognize a single resolved gesture for the
  frame (first hand whose gesture is `one`/`two`/`fist`, else `other`) →
  `SessionController.update(gesture, now)` → pass `active_washer` into
  `FusionEngine.process(...)`.
- Drops the `IdentityClassifier` and skin/ring overlay; `annotate` now shows:
  - A top banner: `Session: You` / `Session: Wife` /
    `Session: none - show 1/2 fingers`.
  - Per hand: its bounding box and the live recognized gesture label.
  - The sink zone (blue) and dish boxes (green + label), unchanged.
- `annotate` signature becomes
  `annotate(frame, config, hands, dishes, active_washer, gesture)`.

### `config.py` (changed)
- Remove `you_profile` and `wife_profile`.
- Remove `identity_distance` from `Thresholds`; add `gesture_hold: float = 1.0`.
- Keep `sink_zone`, `dish_classes`, `dish_conf`, `yolo_model`, `exit_grace`,
  `cooldown`, `iou_match`.

### `calibrate.py` (changed)
- Remove skin capture (`y`/`w`) entirely. Calibration only drags the sink zone
  and writes the config. No skin profiles.

### Removed
- `identity.py` (`IdentityClassifier`, `AMBIGUOUS_MARGIN`) and its tests.
- `ring.py` and its tests (Phase 0 probe, superseded).
- `Hand.region_pixels` skin sampling in `detector.py` (the detector no longer
  needs to sample skin); `Hand.region_pixels` field may remain unused or be
  removed — remove it to keep the domain honest, updating `domain.py` and the
  hand detector/fakes/tests that set it.

## Data flow / attribution detail

Identity is the session's `active_washer`, locked onto a dish when it is in the
sink. Because sessions are stable across a wash, the value at sink-visit time is
the correct attribution. If no session is active when a dish is washed, it exits
`uncertain` (recorded, not counted) — surfaced by the banner so the user knows
to signal.

## Error handling / edge cases

- **No landmarks** (e.g., dish-only frames, or the fake detector in tests):
  `recognize_gesture` returns `"other"`; session state is unchanged.
- **Accidental mid-wash pose:** the `hold_seconds` debounce requires a stable
  ~1s hold before acting; the banner reflects the active washer so a wrong
  switch is visible and re-signalable. Documented limitation.
- **Multiple hands:** the engine resolves one gesture per frame
  (first meaningful gesture wins) to avoid conflicting updates.
- No new storage failure modes; `WashEvent`/`CountStore` shapes unchanged.

## Testing

All detection stays behind dependency-injected fakes; no models load in tests.

- **gesture.py:** crafted landmark sets produce `one`, `two`, `fist`, `other`;
  missing/short landmarks → `other`; orientation variants (hand pointing
  up/down) still classify by the wrist-distance rule.
- **SessionController:** a gesture held < `hold_seconds` does not act; held
  `>= hold_seconds` sets the expected washer; switching gestures resets the
  timer; `fist` clears; repeated same gesture acts once (idempotent).
- **FusionEngine:** with `active_washer="You"`, a sink-visited dish that exits
  counts for You; with `active_washer=None`, it exits `uncertain`; switching the
  active washer between two dishes attributes each correctly; cooldown still
  collapses bursts.
- **engine e2e:** scripted hand (gesture) + dish fakes drive a full session:
  show `one` → wash a dish → it counts for You; `fist` → a later dish is
  `uncertain`.
- **config/calibrate:** config has no skin profiles and a `gesture_hold`;
  calibrate references no skin capture.

## Out of scope / future

- Simultaneous two-person washing.
- Configurable gesture-to-person mapping.
- Auto-ending a session on inactivity (explicit fist only, for now).
