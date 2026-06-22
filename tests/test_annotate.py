import numpy as np

from dishcounter.config import Config, SkinProfile, Zone
from dishcounter.domain import Dish, Hand
from dishcounter.engine import annotate
from dishcounter.identity import IdentityClassifier

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))


def _cfg() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=50, y2=50),
        you_profile=SkinProfile(cr=165.0, cb=110.0),
        wife_profile=SkinProfile(cr=120.0, cb=150.0),
    )


def test_annotate_labels_hands_with_identity_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    ident = IdentityClassifier(
        SkinProfile(cr=165.0, cb=110.0), SkinProfile(cr=120.0, cb=150.0),
        max_distance=60.0,
    )
    hands = [Hand(id=1, bbox=(10, 10, 30, 30), region_pixels=RED)]
    dishes = [Dish(id=1, bbox=(5, 5, 40, 40), label="plate", confidence=0.9)]
    out = annotate(frame, _cfg(), hands, dishes, ident)
    assert out.shape == frame.shape
    # The overlay must draw something (frame started all-zero).
    assert out.any()


def test_annotate_handles_missing_region_and_no_identity():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30), region_pixels=None)]
    out = annotate(frame, _cfg(), hands, [], None)
    assert out.shape == frame.shape
