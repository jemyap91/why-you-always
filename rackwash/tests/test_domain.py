from rackwash.domain import Hand, WashEvent


def test_hand_centroid_is_bbox_center():
    hand = Hand(id=1, bbox=(10, 20, 30, 60))
    assert hand.centroid == (20.0, 40.0)


def test_hand_defaults_are_independent():
    a = Hand(id=1, bbox=(0, 0, 1, 1))
    b = Hand(id=2, bbox=(0, 0, 1, 1))
    a.landmarks.append((0.5, 0.5))
    assert b.landmarks == []  # no shared mutable default


def test_washevent_fields():
    ev = WashEvent(person="You", timestamp=123.0, confidence=0.9, source_id=7)
    assert (ev.person, ev.timestamp, ev.confidence, ev.source_id) == ("You", 123.0, 0.9, 7)


def test_wash_event_has_source_id():
    ev = WashEvent(person="You", timestamp=1.0, confidence=0.5, source_id=7)
    assert ev.source_id == 7


def test_hand_has_no_region_pixels_field():
    hand = Hand(id=1, bbox=(0, 0, 10, 10))
    assert not hasattr(hand, "region_pixels")


def test_dish_centroid_is_bbox_center():
    from rackwash.domain import Dish

    d = Dish(id=None, bbox=(10, 20, 30, 60), label="plate", confidence=0.9)
    assert d.centroid == (20.0, 40.0)
    assert d.label == "plate"
    assert d.confidence == 0.9
