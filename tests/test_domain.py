import numpy as np

from dishcounter.domain import Hand, WashEvent, bgr_to_ycrcb, median_chroma


def test_hand_centroid_is_bbox_center():
    hand = Hand(id=1, bbox=(10, 20, 30, 60))
    assert hand.centroid == (20.0, 40.0)


def test_hand_defaults_are_independent():
    a = Hand(id=1, bbox=(0, 0, 1, 1))
    b = Hand(id=2, bbox=(0, 0, 1, 1))
    a.landmarks.append((0.5, 0.5))
    assert b.landmarks == []  # no shared mutable default


def test_washevent_fields():
    ev = WashEvent(person="You", timestamp=123.0, confidence=0.9, hand_id=7)
    assert (ev.person, ev.timestamp, ev.confidence, ev.hand_id) == ("You", 123.0, 0.9, 7)


def test_bgr_to_ycrcb_pure_gray_has_neutral_chroma():
    # Gray pixel -> Cr and Cb both 128 (neutral)
    _, cr, cb = bgr_to_ycrcb(100.0, 100.0, 100.0)
    assert round(cr) == 128
    assert round(cb) == 128


def test_bgr_to_ycrcb_red_raises_cr():
    _, cr, _ = bgr_to_ycrcb(0.0, 0.0, 255.0)  # pure red
    assert cr > 128


def test_median_chroma_returns_median_over_pixels():
    # Two gray + one red pixel; median pixel is gray -> neutral chroma
    pixels = np.array([[100, 100, 100], [120, 120, 120], [0, 0, 255]], dtype=np.uint8)
    cr, cb = median_chroma(pixels)
    assert round(cr) == 128
    assert round(cb) == 128
