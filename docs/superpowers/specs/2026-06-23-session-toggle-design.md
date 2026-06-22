# Session toggle (repeat-your-number to end) — design

**Date:** 2026-06-23
**Status:** Approved (pre-implementation)

## Problem

Ending a session requires a separate `fist` gesture. The user wants ending to
be the *same* signal as starting: show your number again to end.

## Goal

`SessionController` becomes a per-person toggle:

- Held **`one`** → set active to `You` if not already You, else clear to `None`.
- Held **`two`** → set active to `Wife` if not already Wife, else clear to `None`.
- Showing the *other* number switches directly to that person.
- **`fist` and any other pose do nothing** (they only break the hold streak).

Start = show your number; end = show it again; switch = show the other number.

## Non-goals

- Changing wash counting, cooldown, coasting, or identity attribution.
- Removing `fist` detection from `gesture.py` (a closed hand is still labeled
  `fist` in the overlay; it simply has no effect on the session).

## Design

### `session.py` — `SessionController.update(gesture, now)`
- Meaningful gestures become `("one", "two")` only. Any other value (`fist`,
  `other`, unrecognized) resets the candidate hold streak and returns the
  unchanged `active`.
- Debounce is unchanged: a meaningful gesture must be held `>= hold_seconds` and
  acts **once** per continuous hold (until the recognized gesture changes), so
  holding a number steady does not flip the session on and off.
- On acting:
  - `one`: `active = None if active == "You" else "You"`.
  - `two`: `active = None if active == "Wife" else "Wife"`.

### `engine.py` — overlay banner copy
- No-session banner: `Session: none - show 1 (You) / 2 (Wife), show again to end`.
- Active banner unchanged (`Session: You` / `Session: Wife`).

### Unchanged
`gesture.py` (still recognizes one/two/fist/other), `WashCycleEngine`,
`Engine._resolve_gesture` (it may still surface `fist` to the overlay; the
controller ignores it), config, store.

## Edge cases

- **Held number:** acts once; to toggle again you must drop the gesture and
  re-show it (the "acts once per hold" rule already enforces this).
- **`active == "You"`, show `two`:** switches to `Wife` (two: active≠Wife → Wife).
- **`active == "Wife"`, show `one`:** switches to `You`.
- **`fist` while active:** no effect (session stays).

## Testing (`tests/test_session.py`, rewritten/extended)

- `one` held → `You`; drop, `one` again → `None` (toggle off).
- `two` held → `Wife`; `two` again → `None`.
- `You` active, `two` held → `Wife` (switch).
- A continuously-held `one` acts once: sets `You` and does not revert while held.
- `fist` held → no change to `active`.
- Brief `one` (< hold) → no change.

## Execution

This is a single-function behavior change plus copy/tests/docs — implemented
directly with TDD (no multi-task subagent plan), per user agreement.
