"""Counting rule: one wash = a hand dwells in the sink for >= min_wash seconds,
then leaves it (out of the zone, or out of frame). Credited to the active
session washer; a per-person cooldown collapses two-handed exits."""

from __future__ import annotations

from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Hand, WashEvent


@dataclass
class _Visit:
    enter_time: float | None = None  # start of the current in-sink visit
    last_in_sink: float = 0.0        # last time the hand was seen in the sink


class WashCycleEngine:
    def __init__(self, sink: Zone, min_wash: float = 3.0, cooldown: float = 3.0) -> None:
        self._sink = sink
        self._min_wash = min_wash
        self._cooldown = cooldown
        self._visits: dict[int, _Visit] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self, hands: list[Hand], now: float, active_washer: str | None
    ) -> list[WashEvent]:
        present: set[int] = set()
        events: list[WashEvent] = []
        for hand in hands:
            if hand.id is None:
                continue
            present.add(hand.id)
            visit = self._visits.setdefault(hand.id, _Visit())
            if self._sink.contains(hand.centroid):
                if visit.enter_time is None:
                    visit.enter_time = now
                visit.last_in_sink = now
            else:
                self._close(visit, hand.id, now, active_washer, events)

        for hid in list(self._visits):
            if hid in present:
                continue
            self._close(self._visits[hid], hid, now, active_washer, events)
            del self._visits[hid]
        self._visits = {
            hid: v for hid, v in self._visits.items() if v.enter_time is not None
        }
        return events

    def _close(self, visit: _Visit, hand_id: int, now: float,
               active_washer: str | None, events: list[WashEvent]) -> None:
        if visit.enter_time is None:
            return
        dwell = visit.last_in_sink - visit.enter_time
        visit.enter_time = None  # close the visit regardless of outcome
        if dwell < self._min_wash or active_washer not in ("You", "Wife"):
            return
        last = self._last_person_fire.get(active_washer)
        if last is not None and (now - last) < self._cooldown:
            return
        events.append(
            WashEvent(person=active_washer, timestamp=now, confidence=1.0,
                      source_id=hand_id)
        )
        self._last_person_fire[active_washer] = now
