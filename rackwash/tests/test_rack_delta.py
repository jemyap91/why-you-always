from rackwash.config import RackZone
from rackwash.domain import Dish
from rackwash.rack_delta import RackDeltaCounter

LABELS = {"plate", "bowl", "cup", "glass", "mug"}
R = RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False)
RC = RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=True)


def _plate(cx, cy=50):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label="plate", confidence=0.9)


def _det(dishes):
    return lambda: list(dishes)


def test_session_counts_dishes_added_to_rack():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    assert c.process("You", [], 1.0, _det([])) == []        # finalize start, baseline 0
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det(two))
    c.process(None, [], 10.5, _det(two))
    ev = c.process(None, [], 11.0, _det(two))
    assert len(ev) == 2 and all(e.person == "You" and e.source_id == -1 for e in ev)


def test_no_dishes_added_counts_zero():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    c.process(None, [], 10.0, _det([]))
    c.process(None, [], 10.5, _det([]))
    assert c.process(None, [], 11.0, _det([])) == []


def test_dishes_removed_counts_zero():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    two = [_plate(20), _plate(40)]
    c.process("You", [], 0.0, _det(two))
    c.process("You", [], 0.5, _det(two))
    c.process("You", [], 1.0, _det(two))
    c.process(None, [], 10.0, _det([]))
    c.process(None, [], 10.5, _det([]))
    assert c.process(None, [], 11.0, _det([])) == []


def test_burst_median_ignores_a_flicker_frame():
    c = RackDeltaCounter([R], LABELS, rack_window=1.5, dish_interval=0.4)
    for t in (0.0, 0.5, 1.0, 1.5):
        c.process("You", [], t, _det([]))
    two = [_plate(20), _plate(40)]
    three = [_plate(20), _plate(40), _plate(60)]
    c.process(None, [], 10.0, _det(two))
    c.process(None, [], 10.5, _det(two))
    c.process(None, [], 11.0, _det(three))   # flicker
    ev = c.process(None, [], 11.5, _det(two))
    assert len(ev) == 2


def test_requires_clear_skips_occluded_then_counts_clear_frame():
    c = RackDeltaCounter([RC], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    ev = c.process(None, [], 11.0, _det(two))   # clear frame -> counts
    assert len(ev) == 2


def test_requires_clear_all_occluded_counts_zero():
    c = RackDeltaCounter([RC], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    assert c.process(None, [], 11.0, _det([person, *two])) == []


def test_always_count_rack_counts_despite_person_present():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    person = Dish(id=None, bbox=(0, 0, 100, 100), label="person", confidence=0.9)
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det([person, *two]))
    c.process(None, [], 10.5, _det([person, *two]))
    ev = c.process(None, [], 11.0, _det([person, *two]))
    assert len(ev) == 2


def test_switch_credits_old_washer_and_rebaselines():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))
    one = [_plate(20)]
    c.process("Wife", [], 5.0, _det(one))
    c.process("Wife", [], 5.5, _det(one))
    ev = c.process("Wife", [], 6.0, _det(one))
    assert len(ev) == 1 and ev[0].person == "You"
    three = [_plate(20), _plate(40), _plate(60)]
    c.process(None, [], 10.0, _det(three))
    c.process(None, [], 10.5, _det(three))
    ev = c.process(None, [], 11.0, _det(three))
    assert len(ev) == 2 and all(e.person == "Wife" for e in ev)


def test_detect_called_only_during_bursts_and_throttled():
    calls = []

    def detect():
        calls.append(True)
        return []

    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.5)
    c.process(None, [], 0.0, detect)        # idle -> no call
    assert calls == []
    c.process("You", [], 1.0, detect)       # edge -> call 1
    c.process("You", [], 1.2, detect)       # throttled
    c.process("You", [], 1.6, detect)       # call 2
    c.process("You", [], 2.0, detect)       # finalize; throttled
    c.process("You", [], 3.0, detect)       # active, not collecting
    assert len(calls) == 2


def test_last_summary_reports_start_end_and_delta():
    c = RackDeltaCounter([R], LABELS, rack_window=1.0, dish_interval=0.4)
    c.process("You", [], 0.0, _det([]))
    c.process("You", [], 0.5, _det([]))
    c.process("You", [], 1.0, _det([]))            # finalize start burst
    assert c.last_summary == {"washer": None, "start": [None], "end": [0], "delta": 0}
    two = [_plate(20), _plate(40)]
    c.process(None, [], 10.0, _det(two))
    c.process(None, [], 10.5, _det(two))
    c.process(None, [], 11.0, _det(two))           # finalize end burst
    assert c.last_summary == {"washer": "You", "start": [0], "end": [2], "delta": 2}
