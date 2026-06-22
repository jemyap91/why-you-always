"""Ring-region sampling for identity by jewelry. A silver ring reads as
near-neutral, low-saturation (metallic gray) at the ring-finger base, distinct
from reddish skin. Pure NumPy so it is testable without OpenCV.

Phase 0: these metrics are surfaced on the live overlay to verify the silver
ring separates from a bare hand before a full ring-identity subsystem is built.
"""

from __future__ import annotations

import numpy as np

from dishcounter.domain import median_chroma

# MediaPipe hand landmark indices for the ring finger.
_RING_MCP = 13  # knuckle (base)
_RING_PIP = 14  # first joint
# A ring sits on the proximal phalanx, a bit above the knuckle toward the joint.
_ALONG = 0.4
# Saturation below this (0..1) counts as "metallic/neutral" rather than skin.
METALLIC_SAT = 0.25


def ring_finger_base_px(
    hand, frame_w: int, frame_h: int
) -> tuple[int, int] | None:
    """Pixel location of the ring-finger base from normalized landmarks, or
    None if the hand has no usable landmarks."""
    lm = hand.landmarks
    if not lm or len(lm) <= _RING_PIP:
        return None
    x13, y13 = lm[_RING_MCP]
    x14, y14 = lm[_RING_PIP]
    rx = x13 + _ALONG * (x14 - x13)
    ry = y13 + _ALONG * (y14 - y13)
    return (int(rx * frame_w), int(ry * frame_h))


def ring_metrics(frame: np.ndarray, hand, patch: int = 14
                 ) -> tuple[float, float] | None:
    """(metallic_pct, median_cr) for the ring-finger-base patch, or None if the
    region can't be sampled. metallic_pct is the % of low-saturation pixels;
    median_cr is the BT.601 Cr (skin ~150, neutral silver ~128)."""
    h, w = frame.shape[:2]
    pt = ring_finger_base_px(hand, w, h)
    if pt is None:
        return None
    cx, cy = pt
    r = patch // 2
    crop = frame[max(0, cy - r): cy + r, max(0, cx - r): cx + r]
    if crop.size == 0:
        return None
    arr = crop.reshape(-1, 3).astype(np.float64)  # BGR
    mx = arr.max(axis=1)
    mn = arr.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / mx, 0.0)  # HSV saturation, 0..1
    metallic_pct = float((sat < METALLIC_SAT).mean() * 100.0)
    cr, _cb = median_chroma(arr)
    return (metallic_pct, cr)
