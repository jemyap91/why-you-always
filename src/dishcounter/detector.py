"""Hand detection behind a Protocol. The MediaPipe implementation is the only
file that imports mediapipe; swapping detectors is a one-file change."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from dishcounter.domain import Hand

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        print("Downloading MediaPipe hand landmark model (~8 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Done.")
    return _MODEL_PATH


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
    """Real detector using the MediaPipe Tasks API (mediapipe >= 0.10)."""

    def __init__(self, max_hands: int = 4) -> None:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: PLC0415

        model_path = _ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, frame: np.ndarray) -> list[Hand]:
        import cv2  # noqa: PLC0415

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = self._detector.detect_for_video(mp_image, ts_ms)

        hands: list[Hand] = []
        if not result.hand_landmarks:
            return hands

        for idx, lm_list in enumerate(result.hand_landmarks):
            pts = [(lm.x, lm.y) for lm in lm_list]
            xs = [int(x * w) for x, _ in pts]
            ys = [int(y * h) for _, y in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            score = 0.0
            if result.handedness and idx < len(result.handedness):
                score = result.handedness[idx][0].score
            hands.append(
                Hand(
                    id=None,
                    bbox=bbox,
                    landmarks=pts,
                    confidence=score,
                )
            )
        return hands

