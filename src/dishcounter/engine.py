"""Wires the vision pipeline and owns the capture loop. Dependency-injected
(camera, detector, clock, jpeg encoder) so the whole pipeline runs in tests
with zero hardware."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from dishcounter.config import Config
from dishcounter.domain import WashEvent
from dishcounter.identity import IdentityClassifier
from dishcounter.state import SharedState
from dishcounter.store import CountStore
from dishcounter.tracker import HandTracker
from dishcounter.zones import ZoneEventEngine


def encode_jpeg(frame: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate(frame: np.ndarray, config: Config, hands) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    for zone, color in (
        (config.sink_zone, (255, 0, 0)),
        (config.drying_zone, (0, 255, 0)),
    ):
        cv2.rectangle(out, (zone.x1, zone.y1), (zone.x2, zone.y2), color, 2)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    return out


class Engine:
    def __init__(
        self,
        camera,
        detector,
        config: Config,
        store: CountStore,
        state: SharedState,
        clock: Callable[[], float] = time.time,
        jpeg_encoder: Callable[[np.ndarray], bytes] = encode_jpeg,
        annotator: Callable = annotate,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._running = False

        t = config.thresholds
        self._tracker = HandTracker(iou_threshold=t.iou_match)
        self._identity = IdentityClassifier(
            config.you_profile, config.wife_profile, max_distance=t.identity_distance
        )
        self._zones = ZoneEventEngine(
            config.sink_zone,
            config.drying_zone,
            self._identity,
            presence_window=t.presence_window,
            cooldown=t.cooldown,
        )

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._tracker.update(self._detector.detect(frame))
        events = self._zones.process(hands, now)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands)
        self._state.publish(
            self._encode(annotated), self._store.totals(now), camera_online=True
        )
        return events

    def run(self) -> None:
        self._running = True
        backoff = 0.5
        while self._running:
            frame = self._camera.read()
            if frame is None:
                # FakeCamera exhausted -> stop; real camera offline -> back off.
                if isinstance(getattr(self._camera, "_frames", None), list):
                    break
                self._state.publish(None, self._store.totals(self._clock()), False)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            backoff = 0.5
            self.process_frame(frame, self._clock())

    def stop(self) -> None:
        self._running = False
