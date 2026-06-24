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

    def publish(
        self, frame_jpeg: bytes | None, counts: dict, camera_online: bool
    ) -> None:
        with self._lock:
            self._frame_jpeg = frame_jpeg
            self._counts = copy.deepcopy(counts)
            self._camera_online = camera_online

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame_jpeg": self._frame_jpeg,
                "counts": copy.deepcopy(self._counts),
                "camera_online": self._camera_online,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame_jpeg
