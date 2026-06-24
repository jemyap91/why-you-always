"""Thread-safe handoff between the vision engine and the web layer. The engine
publishes the latest annotated JPEG + counts; the web thread reads snapshots.
The engine never blocks on readers."""

from __future__ import annotations

import copy
import threading


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_jpeg: bytes | None = None
        self._counts: dict = {}
        self._camera_online = False
        self._show_detections = False  # web-toggled; engine reads it
        self._detections: list[str] = []  # dish labels seen this frame (preview)

    def publish(
        self, frame_jpeg: bytes | None, counts: dict, camera_online: bool,
        detections: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._frame_jpeg = frame_jpeg
            self._counts = copy.deepcopy(counts)
            self._camera_online = camera_online
            self._detections = list(detections or [])

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame_jpeg": self._frame_jpeg,
                "counts": copy.deepcopy(self._counts),
                "camera_online": self._camera_online,
                "show_detections": self._show_detections,
                "detections": list(self._detections),
            }

    def toggle_detections(self) -> bool:
        with self._lock:
            self._show_detections = not self._show_detections
            return self._show_detections

    def show_detections(self) -> bool:
        with self._lock:
            return self._show_detections

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg
