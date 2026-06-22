from dishcounter.config import Zone
from dishcounter.domain import Hand
from dishcounter.tracker import IouTracker
from dishcounter.washcycle import WashCycleEngine

SINK = Zone(x1=0, y1=0, x2=100, y2=100)


def _hand(cx, cy, hid):
    return Hand(id=hid, bbox=(cx - 10, cy - 10, cx + 10, cy + 10))


def _in(hid):       # hand in the sink
    return _hand(50, 50, hid)


def _out(hid):      # hand present but outside the sink
    return _hand(50, 200, hid)


def test_dwell_then_leaving_sink_counts_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    ev = eng.process([_out(1)], 3.5, "You")
    assert len(ev) == 1 and ev[0].person == "You" and ev[0].source_id == 1
    assert ev[0].confidence == 1.0


def test_dwell_too_short_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 2.0, "You")
    assert eng.process([_out(1)], 2.5, "You") == []


def test_hand_disappearing_after_dwell_counts():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    assert len(eng.process([], 3.5, "You")) == 1


def test_two_hands_leaving_together_collapse_to_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1), _hand(60, 50, 2)], 0.0, "You")
    eng.process([_in(1), _hand(60, 50, 2)], 3.0, "You")
    assert len(eng.process([], 3.5, "You")) == 1


def test_no_session_counts_nothing():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, None)
    eng.process([_in(1)], 3.0, None)
    assert eng.process([], 3.5, None) == []


def test_re_entering_sink_counts_again_after_cooldown():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    eng.process([], 3.5, "You")                 # first count
    eng.process([_in(1)], 8.0, "You")           # new visit
    eng.process([_in(1)], 11.0, "You")
    assert len(eng.process([], 11.5, "You")) == 1


def test_hand_never_leaving_sink_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 5.0, "You")
    assert eng.process([_in(1)], 10.0, "You") == []


def test_hand_flicker_during_wash_counts_once():
    # Regression for the old overcounting: detection flicker while washing must
    # not spawn multiple counts — tracker coasting keeps one stable hand id.
    tracker = IouTracker(iou_threshold=0.3, coast_seconds=2.0)
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    seq = [(0.0, True), (1.8, False), (2.0, True), (3.8, False),
           (4.0, True), (8.0, False), (10.0, False)]
    count = 0
    for now, present in seq:
        hands = tracker.update([Hand(id=None, bbox=(40, 40, 60, 60))] if present else [], now)
        count += len(eng.process(hands, now, "You"))
    assert count == 1
