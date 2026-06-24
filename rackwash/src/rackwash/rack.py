"""Pure helpers for counting dishware in a rack region and deciding whether a
rack is occluded. No model, no state."""

from __future__ import annotations

from rackwash.config import RackZone
from rackwash.domain import Dish, Hand


def count_dishware(dishes: list[Dish], zone: RackZone, dishware_labels: set[str]) -> int:
    return sum(
        1 for d in dishes if d.label in dishware_labels and zone.contains(d.centroid)
    )


def rack_occluded(zone: RackZone, persons: list[Dish], hands: list[Hand]) -> bool:
    return any(zone.overlaps(p.bbox) for p in persons) or any(
        zone.overlaps(h.bbox) for h in hands
    )
