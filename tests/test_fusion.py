from dishcounter.config import Zone
from dishcounter.domain import Dish
from dishcounter.fusion import FusionEngine

SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _dish(cx, did, in_sink=True):
    cy = 50 if in_sink else 200
    return Dish(id=did, bbox=(cx - 10, cy - 10, cx + 10, cy + 10), label="plate",
                confidence=0.9)


def test_dish_washed_during_a_session_counts_for_active_washer():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    assert eng.process([_dish(50, 1)], now=0.0, active_washer="You") == []
    events = eng.process([], now=2.0, active_washer="You")  # gone > grace
    assert len(events) == 1
    assert events[0].person == "You"
    assert events[0].confidence == 1.0
    assert events[0].source_id == 1


def test_dish_washed_with_no_active_session_is_uncertain():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1)], now=0.0, active_washer=None)
    events = eng.process([], now=2.0, active_washer=None)
    assert len(events) == 1
    assert events[0].person == "uncertain"


def test_switching_active_washer_attributes_each_dish_correctly():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(40, 1)], now=0.0, active_washer="You")
    first = eng.process([], now=2.0, active_washer="You")          # You's dish exits
    eng.process([_dish(60, 2)], now=6.0, active_washer="Wife")     # new dish, Wife active
    second = eng.process([], now=8.0, active_washer="Wife")
    assert [e.person for e in first] == ["You"]
    assert [e.person for e in second] == ["Wife"]


def test_dish_never_in_sink_does_not_count():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1, in_sink=False)], now=0.0, active_washer="You")
    assert eng.process([], now=2.0, active_washer="You") == []


def test_two_dishes_same_session_within_cooldown_collapse_to_one():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(40, 1), _dish(60, 2)], now=0.0, active_washer="You")
    events = eng.process([], now=2.0, active_washer="You")
    assert len(events) == 1


def test_gone_exactly_at_grace_does_not_fire_but_just_past_does():
    eng = FusionEngine(SINK, exit_grace=1.5, cooldown=3.0)
    eng.process([_dish(50, 1)], now=0.0, active_washer="You")
    assert eng.process([], now=1.5, active_washer="You") == []     # == grace, no fire
    assert len(eng.process([], now=1.6, active_washer="You")) == 1  # just past
