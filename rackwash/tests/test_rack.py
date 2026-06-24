from rackwash.config import RackZone
from rackwash.domain import Dish, Hand
from rackwash.rack import count_dishware, rack_occluded

LABELS = {"plate", "bowl", "cup", "glass", "mug"}
ZONE = RackZone(x1=0, y1=0, x2=100, y2=100)


def _dish(cx, cy, label="plate"):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label=label, confidence=0.9)


def test_count_dishware_counts_in_zone_dishware_only():
    dishes = [
        _dish(50, 50, "plate"),    # in zone, dishware -> counts
        _dish(50, 50, "fork"),     # not dishware -> ignored
        _dish(200, 50, "plate"),   # outside zone -> ignored
        _dish(50, 50, "person"),   # not dishware -> ignored
    ]
    assert count_dishware(dishes, ZONE, LABELS) == 1


def test_rack_occluded_true_when_person_box_overlaps():
    person = Dish(id=None, bbox=(80, 80, 160, 160), label="person", confidence=0.9)
    assert rack_occluded(ZONE, [person], []) is True


def test_rack_occluded_true_when_hand_box_overlaps():
    assert rack_occluded(ZONE, [], [Hand(id=1, bbox=(90, 90, 130, 130))]) is True


def test_rack_occluded_false_when_nothing_overlaps():
    person = Dish(id=None, bbox=(200, 200, 260, 260), label="person", confidence=0.9)
    assert rack_occluded(ZONE, [person], [Hand(id=1, bbox=(300, 300, 320, 320))]) is False
