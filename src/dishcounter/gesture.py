"""Recognize simple hand gestures from MediaPipe landmarks. A finger is
'extended' when its tip is farther from the wrist than its mid-joint, which is
orientation-tolerant and needs no calibration. Pure geometry, no frame access."""

from __future__ import annotations

import math

_WRIST = 0
# finger name -> (tip landmark, pip landmark)
_FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}


def _extended_fingers(landmarks: list[tuple[float, float]]) -> set[str]:
    wrist = landmarks[_WRIST]
    out: set[str] = set()
    for name, (tip, pip) in _FINGERS.items():
        if math.dist(landmarks[tip], wrist) > math.dist(landmarks[pip], wrist):
            out.add(name)
    return out


def recognize_gesture(hand) -> str:
    lm = hand.landmarks
    if not lm or len(lm) < 21:
        return "other"
    ext = _extended_fingers(lm)
    if not ext:
        return "fist"
    if ext == {"index"}:
        return "one"
    if ext == {"index", "middle"}:
        return "two"
    return "other"
