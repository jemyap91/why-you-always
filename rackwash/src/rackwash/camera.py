"""Camera sources. FakeCamera drives tests; Camera wraps OpenCV with lazy
open + reconnect so the engine survives a webcam unplug."""

from __future__ import annotations

import numpy as np


class FakeCamera:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self._i = 0

    def read(self) -> np.ndarray | None:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame

    def release(self) -> None:
        self._frames = []


class Camera:
    def __init__(self, index: int = 0, max_backoff: float = 5.0) -> None:
        self._index = index
        self._max_backoff = max_backoff
        self._cap = None

    def _ensure_open(self) -> bool:
        import cv2  # noqa: PLC0415

        if self._cap is not None and self._cap.isOpened():
            return True
        self._cap = cv2.VideoCapture(self._index)
        return bool(self._cap.isOpened())

    def read(self) -> np.ndarray | None:
        if not self._ensure_open():
            return None
        ok, frame = self._cap.read()
        if not ok:
            self.release()  # force reopen next call
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
