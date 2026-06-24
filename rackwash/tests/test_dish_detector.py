import numpy as np

from rackwash.dish_detector import (
    DishDetector,
    FakeDishDetector,
    dishes_from_detections,
)
from rackwash.domain import Dish


def test_fake_dish_detector_replays_script_and_clamps_to_last():
    d = Dish(id=None, bbox=(0, 0, 10, 10), label="plate", confidence=0.9)
    det = FakeDishDetector([[d], []])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert det.detect(frame) == [d]
    assert det.detect(frame) == []
    assert det.detect(frame) == []  # clamps to last script entry
    assert isinstance(det, DishDetector)


def test_dishes_from_detections_filters_low_confidence_and_maps_fields():
    raw = [
        ((0, 0, 20, 20), "plate", 0.8),
        ((5, 5, 9, 9), "fork", 0.2),  # below threshold -> dropped
    ]
    dishes = dishes_from_detections(raw, min_conf=0.4)
    assert len(dishes) == 1
    assert dishes[0].label == "plate"
    assert dishes[0].bbox == (0, 0, 20, 20)
    assert dishes[0].confidence == 0.8
    assert dishes[0].id is None
