from dishcounter.domain import Hand
from dishcounter.tracker import HandTracker, IouTracker, iou


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_tracker_assigns_fresh_ids_to_new_hands():
    tracker = HandTracker()
    hands = [Hand(id=None, bbox=(0, 0, 10, 10)), Hand(id=None, bbox=(50, 50, 60, 60))]
    out = tracker.update(hands)
    assert {h.id for h in out} == {0, 1}


def test_tracker_keeps_id_for_overlapping_hand_across_frames():
    tracker = HandTracker()
    first = tracker.update([Hand(id=None, bbox=(0, 0, 10, 10))])
    original_id = first[0].id
    # Next frame: nearly the same box (high IoU) -> same id.
    second = tracker.update([Hand(id=None, bbox=(1, 1, 11, 11))])
    assert second[0].id == original_id


def test_tracker_gives_new_id_when_hand_jumps_far():
    tracker = HandTracker()
    tracker.update([Hand(id=None, bbox=(0, 0, 10, 10))])
    moved = tracker.update([Hand(id=None, bbox=(200, 200, 210, 210))])
    assert moved[0].id == 1  # not matched to track 0


def test_tracker_does_not_assign_one_track_to_two_hands():
    tracker = HandTracker()
    tracker.update([Hand(id=None, bbox=(0, 0, 10, 10))])
    out = tracker.update(
        [Hand(id=None, bbox=(1, 1, 11, 11)), Hand(id=None, bbox=(2, 2, 12, 12))]
    )
    assert len({h.id for h in out}) == 2  # distinct ids


def test_coasting_reemits_lost_track_then_drops_it():
    tr = IouTracker(iou_threshold=0.3, coast_seconds=2.0)
    h = Hand(id=None, bbox=(0, 0, 40, 40))
    [t0] = tr.update([h], now=0.0)
    first = t0.id
    # Missed frame within the coast window -> the track is re-emitted (ghost).
    coasted = tr.update([], now=1.0)
    assert len(coasted) == 1 and coasted[0].id == first
    coasted2 = tr.update([], now=1.9)
    assert len(coasted2) == 1 and coasted2[0].id == first
    # Beyond the coast window -> dropped.
    assert tr.update([], now=2.5) == []


def test_coasting_reassociates_returning_detection_to_same_id():
    tr = IouTracker(iou_threshold=0.3, coast_seconds=2.0)
    [a] = tr.update([Hand(id=None, bbox=(0, 0, 40, 40))], now=0.0)
    first = a.id
    tr.update([], now=1.0)  # coasting (no detection this frame)
    [b] = tr.update([Hand(id=None, bbox=(5, 5, 45, 45))], now=1.5)
    assert b.id == first  # returning detection re-uses the coasted id


def test_no_coasting_by_default_drops_immediately():
    tr = IouTracker(iou_threshold=0.3)  # coast_seconds defaults to 0
    tr.update([Hand(id=None, bbox=(0, 0, 40, 40))], now=0.0)
    assert tr.update([], now=0.5) == []  # dropped immediately, no ghost
