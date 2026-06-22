import numpy as np

from dishcounter.config import SkinProfile, Zone
from dishcounter.domain import Dish, Hand
from dishcounter.fusion import FusionEngine
from dishcounter.identity import IdentityClassifier

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))   # -> You
BLUE = np.tile(np.array([220, 90, 90], dtype=np.uint8), (16, 1))  # -> Wife
SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _identity():
    return IdentityClassifier(
        SkinProfile(cr=165.0, cb=110.0), SkinProfile(cr=120.0, cb=150.0),
        max_distance=60.0,
    )


def _hand(cx, region, hid):
    return Hand(id=hid, bbox=(cx - 10, 40, cx + 10, 60), region_pixels=region)


def _dish(cx, did, in_sink=True):
    cy = 50 if in_sink else 200
    return Dish(id=did, bbox=(cx - 10, cy - 10, cx + 10, cy + 10), label="plate",
                confidence=0.9)


def test_sink_dish_that_leaves_fires_one_event_for_nearest_hand():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    assert eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0) == []
    events = eng.process([], [], now=2.0)  # dish gone for 2.0s > grace
    assert len(events) == 1
    assert events[0].person == "You"
    assert events[0].source_id == 1


def test_flicker_under_grace_does_not_fire():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0)
    eng.process([], [], now=0.5)                      # brief dropout
    events = eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=1.0)  # back
    events += eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=1.2)
    assert events == []


def test_dish_never_in_sink_does_not_count():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1, in_sink=False)], now=0.0)
    assert eng.process([], [], now=2.0) == []


def test_two_dishes_same_person_within_cooldown_collapse_to_one():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(40, 1), _dish(60, 2)], now=0.0)
    events = eng.process([], [], now=2.0)  # both gone together
    assert len(events) == 1
    assert events[0].person == "You"


def test_dish_washed_with_no_confident_hand_is_uncertain():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([], [_dish(50, 1)], now=0.0)  # no hands near
    events = eng.process([], [], now=2.0)
    assert len(events) == 1
    assert events[0].person == "uncertain"


def test_reappearing_dish_with_new_id_counts_again():
    eng = FusionEngine(SINK, _identity(), exit_grace=1.5, cooldown=3.0)
    eng.process([_hand(50, RED, 1)], [_dish(50, 1)], now=0.0)
    first = eng.process([], [], now=2.0)               # fires You
    eng.process([_hand(50, RED, 2)], [_dish(50, 2)], now=6.0)  # new dish id
    second = eng.process([], [], now=8.0)              # past cooldown -> fires
    assert len(first) == 1 and len(second) == 1
