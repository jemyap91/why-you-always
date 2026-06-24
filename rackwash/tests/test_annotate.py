import numpy as np

from rackwash.config import Config, RackZone
from rackwash.domain import Hand
from rackwash.engine import annotate


def _cfg() -> Config:
    return Config(
        camera_index=0,
        rack_zones=[RackZone(x1=0, y1=0, x2=50, y2=50, requires_clear=False)],
    )


def test_annotate_runs_and_preserves_shape():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    out = annotate(frame, _cfg(), [Hand(id=1, bbox=(10, 10, 30, 30))], "You", "one")
    assert out.shape == frame.shape
    assert out.any()


def test_annotate_no_session_or_hands():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    assert annotate(frame, _cfg(), [], None, "other").shape == frame.shape


def test_annotate_collecting_marker_adds_pixels():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    base = annotate(frame, _cfg(), [], "You", "other", False)
    marked = annotate(frame, _cfg(), [], "You", "other", True)
    assert int(marked.sum()) > int(base.sum())
