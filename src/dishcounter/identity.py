"""Classify a hand region's median skin chroma against two calibrated profiles.
Biased toward 'uncertain' — we never mis-attribute a wash."""

from __future__ import annotations

import math

from dishcounter.config import SkinProfile
from dishcounter.domain import Hand, median_chroma

AMBIGUOUS_MARGIN = 5.0  # min gap between the two distances to commit


class IdentityClassifier:
    def __init__(
        self, you: SkinProfile, wife: SkinProfile, max_distance: float = 25.0
    ) -> None:
        self._you = you
        self._wife = wife
        self._max_distance = max_distance

    def classify(self, hand: Hand) -> tuple[str, float]:
        if hand.region_pixels is None or len(hand.region_pixels) == 0:
            return ("uncertain", 0.0)

        cr, cb = median_chroma(hand.region_pixels)
        d_you = math.dist((cr, cb), (self._you.cr, self._you.cb))
        d_wife = math.dist((cr, cb), (self._wife.cr, self._wife.cb))
        nearer, farther = sorted((d_you, d_wife))

        if nearer > self._max_distance:
            return ("uncertain", 0.0)
        if (farther - nearer) < AMBIGUOUS_MARGIN:
            return ("uncertain", 0.0)

        label = "You" if d_you < d_wife else "Wife"
        confidence = max(0.0, min(1.0, 1.0 - nearer / self._max_distance))
        return (label, confidence)
