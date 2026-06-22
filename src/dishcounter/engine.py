"""Wires the vision pipeline and owns the capture loop. Dependency-injected
(camera, detector, clock, jpeg encoder) so the whole pipeline runs in tests
with zero hardware."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from dishcounter.config import Config
from dishcounter.dish_detector import DishDetector
from dishcounter.domain import WashEvent
from dishcounter.fusion import FusionEngine
from dishcounter.gesture import recognize_gesture
from dishcounter.session import SessionController
from dishcounter.state import SharedState
from dishcounter.store import CountStore
from dishcounter.tracker import IouTracker


def encode_jpeg(frame: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate(frame: np.ndarray, config: Config, hands, dishes,
             active_washer, gesture) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    z = config.sink_zone
    cv2.rectangle(out, (z.x1, z.y1), (z.x2, z.y2), (255, 0, 0), 2)
    for dish in dishes:
        x1, y1, x2, y2 = dish.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, dish.label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
    if active_washer:
        banner, color = f"Session: {active_washer}", (0, 255, 0)
    else:
        banner = "Session: none - show 1 finger (You) / 2 (Wife)"
        color = (0, 165, 255)
    cv2.putText(out, banner, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"gesture: {gesture}", (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


class Engine:
    def __init__(
        self,
        camera,
        detector,
        dish_detector: DishDetector,
        config: Config,
        store: CountStore,
        state: SharedState,
        clock: Callable[[], float] = time.time,
        jpeg_encoder: Callable[[np.ndarray], bytes] = encode_jpeg,
        annotator: Callable = annotate,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._dish_detector = dish_detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._running = False

        t = config.thresholds
        self._hand_tracker = IouTracker(iou_threshold=t.iou_match)
        self._dish_tracker = IouTracker(iou_threshold=t.iou_match)
        self._session = SessionController(hold_seconds=t.gesture_hold)
        self._fusion = FusionEngine(
            config.sink_zone, exit_grace=t.exit_grace, cooldown=t.cooldown
        )

    @staticmethod
    def _resolve_gesture(hands) -> str:
        for hand in hands:
            g = recognize_gesture(hand)
            if g in ("one", "two", "fist"):
                return g
        return "other"

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame))
        dishes = self._dish_tracker.update(self._dish_detector.detect(frame))
        gesture = self._resolve_gesture(hands)
        active = self._session.update(gesture, now)
        events = self._fusion.process(dishes, now, active)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands, dishes, active, gesture)
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
