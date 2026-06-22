"""The counting rule. Dishes drive the count: a dish that was washed at the
sink and then disappears from view is one washed dish. Hands only supply the
You/Wife identity (nearest hand at the sink)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Dish, Hand, WashEvent
from dishcounter.identity import IdentityClassifier


@dataclass
class _DishState:
    last_sink_time: float | None = None
    last_seen: float = 0.0
    locked_person: str = "uncertain"
    locked_confidence: float = 0.0


class FusionEngine:
    def __init__(
        self,
        sink: Zone,
        identity: IdentityClassifier,
        exit_grace: float = 1.5,
        cooldown: float = 3.0,
    ) -> None:
        self._sink = sink
        self._identity = identity
        self._exit_grace = exit_grace
        self._cooldown = cooldown
        self._states: dict[int, _DishState] = {}
        self._last_person_fire: dict[str, float] = {}

    def process(
        self, hands: list[Hand], dishes: list[Dish], now: float
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
                if state.locked_person == "uncertain":
                    self._lock_identity(state, dish, hands)

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

    def _lock_identity(self, state: _DishState, dish: Dish, hands: list[Hand]
                       ) -> None:
        hand = self._nearest_hand(dish, hands)
        if hand is None:
            return
        label, conf = self._identity.classify(hand)
        if label != "uncertain":
            state.locked_person = label
            state.locked_confidence = conf

    @staticmethod
    def _nearest_hand(dish: Dish, hands: list[Hand]) -> Hand | None:
        if not hands:
            return None
        return min(hands, key=lambda h: math.dist(h.centroid, dish.centroid))

    def _can_fire(self, person: str, now: float) -> bool:
        if person not in ("You", "Wife"):
            return True  # uncertain: recorded, no cooldown
        last = self._last_person_fire.get(person)
        return last is None or (now - last) >= self._cooldown
