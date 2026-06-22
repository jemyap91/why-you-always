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
