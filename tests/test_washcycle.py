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


def test_dwell_with_dish_seen_counts_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    ev = eng.process([_out(1)], 3.5, "You")
    assert len(ev) == 1 and ev[0].person == "You" and ev[0].source_id == 1
    assert ev[0].confidence == 1.0


def test_dwell_without_any_dish_seen_does_not_count():
    # The fix: a hand dwell-and-leave with no dish in the sink is NOT a wash.
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You")          # dish_seen defaults False
    eng.process([_in(1)], 3.0, "You")
    assert eng.process([_out(1)], 3.5, "You") == []


def test_dish_min_hits_zero_disables_the_gate():
    # Escape hatch: behaves like the old hand-only counter.
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0, dish_min_hits=0)
    eng.process([_in(1)], 0.0, "You")
    eng.process([_in(1)], 3.0, "You")
    assert len(eng.process([_out(1)], 3.5, "You")) == 1


def test_dwell_too_short_does_not_count_even_with_dish():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 2.0, "You", dish_seen=True)
    assert eng.process([_out(1)], 2.5, "You") == []


def test_hand_disappearing_after_dish_dwell_counts():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    assert len(eng.process([], 3.5, "You")) == 1


def test_two_hands_leaving_together_collapse_to_one():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1), _hand(60, 50, 2)], 0.0, "You", dish_seen=True)
    eng.process([_in(1), _hand(60, 50, 2)], 3.0, "You", dish_seen=True)
    assert len(eng.process([], 3.5, "You")) == 1


def test_no_session_counts_nothing():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, None, dish_seen=True)
    eng.process([_in(1)], 3.0, None, dish_seen=True)
    assert eng.process([], 3.5, None) == []


def test_re_entry_needs_its_own_dish_confirmation():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    eng.process([], 3.5, "You")                      # first count (dish_hits reset)
    eng.process([_in(1)], 8.0, "You")                # new visit, NO dish this time
    eng.process([_in(1)], 11.0, "You")
    assert eng.process([], 11.5, "You") == []         # not counted: no dish seen


def test_hand_never_leaving_sink_does_not_count():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 5.0, "You", dish_seen=True)
    assert eng.process([_in(1)], 10.0, "You", dish_seen=True) == []


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
        count += len(eng.process(hands, now, "You", dish_seen=True))
    assert count == 1


def test_idle_out_of_sink_hands_do_not_accumulate_state():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    for i in range(50):
        eng.process([_out(i)], float(i), "You")
    assert eng._visits == {}  # no dead entries left behind


def test_re_entry_with_dish_counts_again_after_cooldown():
    eng = WashCycleEngine(SINK, min_wash=3.0, cooldown=3.0)
    eng.process([_in(1)], 0.0, "You", dish_seen=True)
    eng.process([_in(1)], 3.0, "You", dish_seen=True)
    eng.process([], 3.5, "You")                          # first count
    eng.process([_in(1)], 8.0, "You", dish_seen=True)    # new visit WITH a dish
    eng.process([_in(1)], 11.0, "You", dish_seen=True)
    assert len(eng.process([], 11.5, "You")) == 1         # counts again past cooldown
