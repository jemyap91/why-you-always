from dishcounter.domain import Hand
from dishcounter.tracker import HandTracker, iou


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
