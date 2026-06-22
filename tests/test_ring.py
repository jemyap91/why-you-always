import numpy as np

from dishcounter.domain import Hand
from dishcounter.ring import ring_finger_base_px, ring_metrics


def _hand_with_ring_landmark_at(fx: float, fy: float) -> Hand:
    # 21 normalized landmarks; ring MCP (13) and PIP (14) both at (fx, fy) so
    # the sampled base point lands exactly there.
    lm = [(0.5, 0.5)] * 21
    lm[13] = (fx, fy)
    lm[14] = (fx, fy)
    return Hand(id=1, bbox=(0, 0, 10, 10), landmarks=lm)


def test_ring_finger_base_none_without_landmarks():
    bare = Hand(id=1, bbox=(0, 0, 10, 10), landmarks=[])
    assert ring_finger_base_px(bare, 100, 100) is None
    assert ring_metrics(np.zeros((100, 100, 3), dtype=np.uint8), bare) is None


def test_silver_patch_reads_metallic_and_neutral():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (190, 190, 190)  # gray (B=G=R) -> zero saturation, neutral Cr
    hand = _hand_with_ring_landmark_at(0.5, 0.5)
    metallic_pct, cr = ring_metrics(frame, hand)
    assert metallic_pct == 100.0          # all pixels low-saturation
    assert abs(cr - 128.0) < 1.0          # neutral chroma


def test_skin_patch_reads_non_metallic_and_reddish():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (90, 110, 200)  # BGR reddish skin -> saturated, elevated Cr
    hand = _hand_with_ring_landmark_at(0.5, 0.5)
    metallic_pct, cr = ring_metrics(frame, hand)
    assert metallic_pct == 0.0            # no low-saturation pixels
    assert cr > 140.0                     # clearly reddish
