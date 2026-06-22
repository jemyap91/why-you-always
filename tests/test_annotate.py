import numpy as np

from dishcounter.config import Config, Zone
from dishcounter.domain import Dish, Hand
from dishcounter.engine import annotate


def _cfg() -> Config:
    return Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=50, y2=50))


def test_annotate_runs_with_active_session_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]
    dishes = [Dish(id=1, bbox=(5, 5, 40, 40), label="plate", confidence=0.9)]
    out = annotate(frame, _cfg(), hands, dishes, "You", "one")
    assert out.shape == frame.shape
    assert out.any()  # something was drawn on the all-zero frame


def test_annotate_runs_with_no_session():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [], [], None, "other")
    assert out.shape == frame.shape
