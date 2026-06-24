"""Outcome-based counting: at each session boundary, run a short burst of dish
counts over the rack regions and credit the net increase to the washer whose
session just closed. YOLO runs only during bursts (via the injected `detect`
callback). A requires_clear rack is sampled only on frames where no person/hand
overlaps it; a rack only counts when measured at BOTH boundaries."""

from __future__ import annotations

import statistics
from collections.abc import Callable

from rackwash.config import RackZone
from rackwash.domain import Dish, Hand, WashEvent
from rackwash.rack import count_dishware, rack_occluded


class RackDeltaCounter:
    def __init__(
        self,
        rack_zones: list[RackZone],
        dishware_labels: set[str],
        rack_window: float = 1.5,
        dish_interval: float = 0.5,
    ) -> None:
        self._zones = list(rack_zones)
        self._labels = set(dishware_labels)
        self._window = rack_window
        self._interval = dish_interval
        n = len(self._zones)
        self._washer: str | None = None
        self._baseline: list[int | None] = [None] * n
        self._prev_active: str | None = None
        self._collecting = False
        self._burst_end = 0.0
        self._last_run: float | None = None
        self._samples: list[list[int]] = [[] for _ in self._zones]
        self._closing_washer: str | None = None
        self._closing_baseline: list[int | None] = []
        self._new_washer: str | None = None

    @property
    def is_collecting(self) -> bool:
        return self._collecting

    def process(
        self,
        active_washer: str | None,
        hands: list[Hand],
        now: float,
        detect: Callable[[], list[Dish]] | None,
    ) -> list[WashEvent]:
        if active_washer != self._prev_active:
            self._start_burst(active_washer, now)
        self._prev_active = active_washer

        events: list[WashEvent] = []
        if self._collecting:
            self._maybe_sample(hands, now, detect)
            if now >= self._burst_end:
                events = self._finalize(now)
        return events

    def _start_burst(self, active_washer: str | None, now: float) -> None:
        if not self._collecting:
            self._collecting = True
            self._closing_washer = self._washer
            self._closing_baseline = list(self._baseline)
            self._samples = [[] for _ in self._zones]
            self._last_run = None
        self._new_washer = active_washer  # coalesce: latest active wins
        self._burst_end = now + self._window

    def _maybe_sample(
        self, hands: list[Hand], now: float, detect: Callable[[], list[Dish]] | None
    ) -> None:
        if detect is None:
            return
        if self._last_run is not None and (now - self._last_run) < self._interval:
            return
        self._last_run = now
        try:
            dishes = detect()
        except Exception:  # noqa: BLE001 - a flaky detector must not crash the loop
            return
        persons = [d for d in dishes if d.label == "person"]
        for i, zone in enumerate(self._zones):
            if zone.requires_clear and rack_occluded(zone, persons, hands):
                continue
            self._samples[i].append(count_dishware(dishes, zone, self._labels))

    def _finalize(self, now: float) -> list[WashEvent]:
        end = [statistics.median_low(s) if s else None for s in self._samples]
        events: list[WashEvent] = []
        if self._closing_washer in ("You", "Wife"):
            delta = 0
            for base, fin in zip(self._closing_baseline, end):
                if base is not None and fin is not None:
                    delta += max(0, fin - base)
            events = [
                WashEvent(
                    person=self._closing_washer, timestamp=now,
                    confidence=1.0, source_id=-1,
                )
                for _ in range(delta)
            ]
        self._baseline = end
        self._washer = self._new_washer
        self._collecting = False
        return events
