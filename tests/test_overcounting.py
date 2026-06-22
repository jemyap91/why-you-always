"""Regression: detection flicker on a dish being washed must not spawn multiple
counts. Tracker coasting bridges the gaps so one physical dish counts once."""

from dishcounter.config import Zone
from dishcounter.domain import Dish
from dishcounter.fusion import FusionEngine
from dishcounter.tracker import IouTracker

SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _det() -> Dish:
    # A fresh detection each frame (id assigned by the tracker), centroid in sink.
    return Dish(id=None, bbox=(40, 40, 60, 60), label="plate", confidence=0.9)


def _run(coast_seconds: float) -> int:
    tracker = IouTracker(iou_threshold=0.3, coast_seconds=coast_seconds)
    fusion = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    # (now, dish_detected): one dish flickers with ~1.8s gaps, then is carried away.
    seq = [(0.0, True), (1.8, False), (2.0, True), (3.8, False),
           (4.0, True), (5.8, False), (6.0, True), (8.0, False), (9.6, False)]
    count = 0
    for now, present in seq:
        dishes = tracker.update([_det()] if present else [], now)
        count += sum(1 for e in fusion.process(dishes, now, "You") if e.person == "You")
    return count


def test_flicker_during_wash_counts_once_with_coasting():
    assert _run(coast_seconds=2.0) == 1


def test_without_coasting_the_same_flicker_overcounts():
    # Documents the bug: with no coasting, flicker spawns more than one count.
    assert _run(coast_seconds=0.0) > 1
