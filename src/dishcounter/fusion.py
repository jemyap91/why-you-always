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
