# Dashboard reset + live dish detections (rackwash) — design

**Date:** 2026-06-24
**Status:** Approved (pre-implementation)
**Target package:** `rackwash`.

## Problem / Goal

Two dashboard additions:

1. **Reset button** — clear the scoreboard (today + all-time) back to zero.
2. **Live dish detections (toggle)** — see, in real time, what dishware the model
   is picking up, to verify detection. rackwash runs YOLO only in boundary
   bursts; live view must be opt-in to preserve that efficiency.

## Feature 1 — Reset

### `store.py` (changed)
- Add `reset(now)`: append a **reset-marker event** (`person="__reset__"`,
  `confidence=0.0`, `counted=0`). Non-destructive — real rows stay in the log.
- `totals`/`_tally` count only events **at or after the latest reset marker**:
  - `floor = MAX(timestamp) WHERE person='__reset__'` (else `0.0`).
  - `all_time` = counted You/Wife events with `timestamp >= floor`.
  - `today` = counted events with `timestamp >= max(day_start, floor)` and
    `< day_end`.
- Add a `threading.Lock`; public `record`/`totals`/`reset` acquire it (private
  `_tally`/`_latest_reset`/`_day_bounds` do not — avoids re-entrancy). This is
  the first time the **web thread writes** to the store; until now only the
  engine thread did.

### `dashboard.py` (changed)
- `create_app(state, store=None)`; add `POST /reset` → `store.reset(time.time())`
  when a store is provided (no-op/200 otherwise).
- A **Reset** button → JS `confirm("Reset all counts?")` → `fetch('/reset',{method:'POST'})`.
- Counts show 0 on the next published frame (the engine republishes `totals`
  every frame).

### `run_server` / `cli.py` (changed)
- `run_server(state, store=None, host=..., port=...)` → `create_app(state, store)`.
- `cli._serve` passes its `CountStore` into `run_server`.

## Feature 2 — Live dish detections (toggle, default off)

### `state.py` (changed)
- Add `show_detections: bool` (default `False`), toggled by the web thread:
  `toggle_detections()` / `show_detections()` getter (lock-guarded).
- `publish(...)` gains an optional `detections: list[str] | None` (labels of
  dishes seen this frame); `snapshot()` returns `show_detections` and
  `detections`.

### `engine.py` (changed)
- `process_frame`: when `state.show_detections()` is true and a dish detector is
  present, run `dish_detector.detect(frame)` **throttled to `dish_interval`**
  (separate `_last_preview_run` clock); otherwise no live detection. Pass the
  dishes to `annotate` and publish their labels.
- `annotate(..., collecting=False, dishes=None)`: when `dishes` given, draw each
  dish **box + label** (e.g. `plate 0.82`) in a distinct color.
- This is inspection-only: it never feeds the `RackDeltaCounter`; counting is
  unchanged (still boundary bursts).

### `dashboard.py` (changed)
- A **Show detections** button → `POST /detections` → `state.toggle_detections()`.
- The WebSocket payload gains `show_detections` and `detections`; the page shows
  a small `detected: plate, cup` line when the toggle is on.

## Edge cases / errors

- `POST /reset` with no store wired → 200, no-op (keeps tests/headless simple).
- Live detector raising → caught; treated as no dishes that frame (don't crash
  the loop).
- Toggle resets to **off** on restart (not persisted) — acceptable.

## Testing

All detection behind dependency-injected fakes; no models in tests.

- **store:** after `reset(t)`, `totals` shows `{You:0, Wife:0}` for today and
  all-time; events recorded after `t` count again; the `__reset__` marker is
  never counted; lock present (smoke: concurrent-safe call path).
- **state:** `show_detections` defaults False, toggles; `publish`/`snapshot`
  round-trip `detections`.
- **engine:** with the toggle on and a fake dish detector, `process_frame`
  publishes detected labels and the annotator receives dishes; with it off, no
  live detection runs (fake detector not called).
- **dashboard:** `POST /reset` calls a store's `reset`; `POST /detections`
  flips the flag; index HTML contains both buttons; WS includes the new fields.

## Out of scope

- Persisting the detections toggle across restarts.
- Per-person reset / undo UI (history remains recoverable in the DB).
- Any change to rack-delta counting or the legacy `dishcounter` package.
