"""Pure helpers for counting dishware in a rack region and deciding whether a
rack is occluded. No model, no state."""

from __future__ import annotations

from rackwash.config import RackZone
from rackwash.domain import Dish, Hand
from rackwash.tracker import iou

# Boxes overlapping a higher-confidence box by more than this IoU are treated as
# the same physical object (the detector often labels one item as both mug+cup).
DEDUPE_IOU = 0.5


def dedupe_overlapping(dishes: list[Dish], iou_threshold: float = DEDUPE_IOU) -> list[Dish]:
    """Class-agnostic non-max suppression: keep the highest-confidence box and
    drop any later box overlapping a kept one by more than iou_threshold, so a
    single object detected under multiple labels counts once."""
    kept: list[Dish] = []
    for d in sorted(dishes, key=lambda x: x.confidence, reverse=True):
        if all(iou(d.bbox, k.bbox) <= iou_threshold for k in kept):
            kept.append(d)
    return kept


def count_dishware(dishes: list[Dish], zone: RackZone, dishware_labels: set[str]) -> int:
    return sum(
        1 for d in dishes if d.label in dishware_labels and zone.contains(d.centroid)
    )


def rack_occluded(zone: RackZone, persons: list[Dish], hands: list[Hand]) -> bool:
    return any(zone.overlaps(p.bbox) for p in persons) or any(
        zone.overlaps(h.bbox) for h in hands
    )
