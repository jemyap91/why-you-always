"""Hand detection behind a Protocol. The MediaPipe implementation is the only
file that imports mediapipe; swapping detectors is a one-file change."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from dishcounter.domain import Hand


@runtime_checkable
class HandDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Hand]: ...


class FakeHandDetector:
    """Deterministic detector for tests/headless runs."""

    def __init__(self, script: list[list[Hand]]) -> None:
        if not script:
            script = [[]]
        self._script = script
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Hand]:
        hands = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return hands


class MediaPipeHandDetector:
    """Real detector. Lazy-imports mediapipe so the rest of the suite runs
    without it installed."""

    def __init__(self, max_hands: int = 4, region_size: int = 24) -> None:
        import mediapipe as mp  # noqa: PLC0415  (lazy by design)

        self._region_size = region_size
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> list[Hand]:
        import cv2  # noqa: PLC0415

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        hands: list[Hand] = []
        if not result.multi_hand_landmarks:
            return hands

        confidences = self._handedness_scores(result)
        for idx, lm in enumerate(result.multi_hand_landmarks):
            pts = [(p.x, p.y) for p in lm.landmark]
            xs = [int(x * w) for x, _ in pts]
            ys = [int(y * h) for _, y in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            region = self._sample_region(frame, int(pts[0][0] * w), int(pts[0][1] * h))
            hands.append(
                Hand(
                    id=None,
                    bbox=bbox,
                    landmarks=pts,
                    region_pixels=region,
                    confidence=confidences[idx] if idx < len(confidences) else 0.0,
                )
            )
        return hands

    def _sample_region(self, frame: np.ndarray, cx: int, cy: int) -> np.ndarray:
        r = self._region_size // 2
        h, w = frame.shape[:2]
        x1, x2 = max(0, cx - r), min(w, cx + r)
        y1, y2 = max(0, cy - r), min(h, cy + r)
        crop = frame[y1:y2, x1:x2]
        return crop.reshape(-1, 3) if crop.size else np.empty((0, 3), dtype=np.uint8)

    @staticmethod
    def _handedness_scores(result) -> list[float]:
        if not result.multi_handedness:
            return []
        return [h.classification[0].score for h in result.multi_handedness]
