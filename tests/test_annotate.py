import numpy as np

from dishcounter.config import Config, Zone
from dishcounter.domain import Hand
from dishcounter.engine import annotate


def _cfg() -> Config:
    return Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=50, y2=50))


def test_annotate_runs_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]  # centroid (20,20) -> in sink
    out = annotate(frame, _cfg(), hands, "You", "one")
    assert out.shape == frame.shape
    assert out.any()


def test_annotate_runs_with_no_session_or_hands():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [], None, "other")
    assert out.shape == frame.shape


def test_annotate_with_dish_seen_draws_marker():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    hands = [Hand(id=1, bbox=(10, 10, 30, 30))]
    base = annotate(frame, _cfg(), hands, "You", "one", False)
    marked = annotate(frame, _cfg(), hands, "You", "one", True)
    assert marked.shape == frame.shape
    # The dish marker adds pixels the un-marked frame does not have.
    assert int(marked.sum()) > int(base.sum())
