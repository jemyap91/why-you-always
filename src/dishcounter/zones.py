"""The counting rule. Tracks each hand id's most recent sink visit and locked
identity, and fires a WashEvent when it crosses into the drying zone."""

from __future__ import annotations

from dataclasses import dataclass

from dishcounter.config import Zone
from dishcounter.domain import Hand, WashEvent
from dishcounter.identity import IdentityClassifier


@dataclass
class _HandState:
    last_sink_time: float | None = None
    locked_person: str = "uncertain"
    locked_confidence: float = 0.0
    last_fire_time: float | None = None


class ZoneEventEngine:
    def __init__(
        self,
        sink: Zone,
        drying: Zone,
        identity: IdentityClassifier,
        presence_window: float = 5.0,
        cooldown: float = 3.0,
    ) -> None:
        self._sink = sink
        self._drying = drying
        self._identity = identity
        self._presence_window = presence_window
        self._cooldown = cooldown
        self._states: dict[int, _HandState] = {}

    def process(self, hands: list[Hand], now: float) -> list[WashEvent]:
        events: list[WashEvent] = []
        for hand in hands:
            if hand.id is None:
                continue
            state = self._states.setdefault(hand.id, _HandState())
            centroid = hand.centroid

            if self._sink.contains(centroid):
                state.last_sink_time = now
                # Lock the first confident identity seen during the sink phase.
                if state.locked_person == "uncertain":
                    label, conf = self._identity.classify(hand)
                    if label != "uncertain":
                        state.locked_person = label
                        state.locked_confidence = conf

            elif self._drying.contains(centroid):
                if self._should_fire(state, now):
                    events.append(
                        WashEvent(
                            person=state.locked_person,
                            timestamp=now,
                            confidence=state.locked_confidence,
                            source_id=hand.id,
                        )
                    )
                    state.last_fire_time = now
                    # Reset the visit so a single crossing counts once.
                    state.last_sink_time = None
                    state.locked_person = "uncertain"
                    state.locked_confidence = 0.0
        return events

    def _should_fire(self, state: _HandState, now: float) -> bool:
        if state.last_sink_time is None:
            return False
        if (now - state.last_sink_time) > self._presence_window:
            return False
        if (
            state.last_fire_time is not None
            and (now - state.last_fire_time) < self._cooldown
        ):
            return False
        return True
