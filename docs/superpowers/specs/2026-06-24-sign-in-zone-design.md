# Sign-in zone for session marking (rackwash) — design

**Date:** 2026-06-24
**Status:** Approved (pre-implementation)
**Target package:** `rackwash` (the live project; `dishcounter` is legacy).

## Problem

In `rackwash`, gesture-based session marking switches/ends by accident.
`Engine._resolve_gesture` reads `1`/`2` from **any** detected hand (no zone
restriction at all — see `test_gesture_registers_anywhere`). During washing,
incidental `1`/`2` finger shapes on any hand toggle/switch the session via the
`SessionController` toggle rule. It is too fragile.

## Goal

Recognize a gesture **only when a hand is inside a small dedicated "sign-in"
zone** — a box placed in a corner, away from the racks and normal hand paths.
Everywhere else, finger poses are ignored. Marking becomes deliberate.

## Approach

Add an optional `sign_in_zone` to the rackwash `Config`. `_resolve_gesture`
accepts a gesture only from a hand whose centroid is inside that zone. The
existing `SessionController` toggle semantics are unchanged.

**Optional, not required:** `sign_in_zone` defaults to `None`. When `None`, the
current "any hand" behavior is preserved (backward compatible — old `config.yaml`
and the existing test that asserts "registers anywhere" keep working). When set
(via re-running `calibrate`), gestures are gated to the zone — which is the fix.
The user must re-run `calibrate` to draw the box and activate it.

## Components

### `config.py` (changed)
- Add `sign_in_zone: Zone | None = None` to `Config`. Serializes to YAML as a
  zone dict or `null`; round-trips through `save`/`load`.

### `engine.py` (changed)
- `_resolve_gesture(hands)`: if `self._config.sign_in_zone` is set, skip any hand
  whose centroid is **not** inside it; then return the first `one`/`two` gesture.
  If `sign_in_zone is None`, behave as today (any hand).
- `annotate(...)`: if `sign_in_zone` is set, draw its rectangle in a distinct
  color (e.g. yellow) with a "sign in" label. Signature unchanged (reads
  `config.sign_in_zone`).

### `calibrate.py` (changed)
- After the rack-zone loop, prompt to drag the **sign-in box** ("drag the
  sign-in box — a corner away from the racks; ESC to skip"). If drawn, save it as
  `sign_in_zone`; if skipped (ESC / zero-size), leave it `None`. Add a pure
  helper `sign_in_zone_from_drag(start, end) -> Zone`.

### Unchanged
`SessionController`, `RackDeltaCounter`, `gesture.py`, store, dashboard.

## Edge cases

- **`sign_in_zone` overlapping a rack or a hand path:** reintroduces false
  triggers; placement is the user's responsibility (prompt + README call it out).
- **`sign_in_zone is None`:** legacy any-hand behavior (back-compat).
- **No hand in the zone:** `_resolve_gesture` returns `"other"`; session unchanged.

## Testing

All detection behind dependency-injected fakes; no models in tests.

- `_resolve_gesture` with a `sign_in_zone` set: a `one`/`two` hand **inside** the
  zone resolves to that gesture; the same gesture **outside** resolves to
  `"other"`.
- Back-compat: with `sign_in_zone=None`, a gesture still registers anywhere
  (existing `test_gesture_registers_anywhere` stays green).
- `config`: `sign_in_zone` defaults to `None` and round-trips through YAML (set
  and unset).
- `annotate`: with a `sign_in_zone` set, the overlay draws extra pixels vs none.
- `calibrate`: `sign_in_zone_from_drag` normalizes corners to a `Zone`.

## Out of scope

- Dashboard buttons for marking (the sign-in zone was chosen instead).
- The parked reset button.
- Changing the rack-delta counting model.
- Any change to the legacy `dishcounter` package.
