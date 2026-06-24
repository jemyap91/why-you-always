import numpy as np

from rackwash.detector import FakeHandDetector, HandDetector
from rackwash.domain import Hand


def test_fake_detector_satisfies_protocol():
    det = FakeHandDetector(script=[[]])
    assert isinstance(det, HandDetector)  # runtime_checkable Protocol


def test_fake_detector_yields_scripted_frames_in_order():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    h1 = Hand(id=None, bbox=(0, 0, 5, 5))
    h2 = Hand(id=None, bbox=(5, 5, 9, 9))
    det = FakeHandDetector(script=[[h1], [h2]])
    assert det.detect(frame) == [h1]
    assert det.detect(frame) == [h2]


def test_fake_detector_repeats_last_frame_when_exhausted():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    h = Hand(id=None, bbox=(0, 0, 5, 5))
    det = FakeHandDetector(script=[[h]])
    det.detect(frame)
    assert det.detect(frame) == [h]  # still returns last entry
