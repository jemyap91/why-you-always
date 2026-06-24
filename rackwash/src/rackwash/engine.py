"""Wires the vision pipeline and owns the capture loop. Dependency-injected so
the whole pipeline runs in tests with zero hardware. Counting is outcome-based:
a RackDeltaCounter counts dishware added to the racks between gesture-marked
session boundaries."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from rackwash.config import Config
from rackwash.domain import WashEvent
from rackwash.gesture import recognize_gesture
from rackwash.rack_delta import RackDeltaCounter
from rackwash.session import SessionController
from rackwash.state import SharedState
from rackwash.store import CountStore
from rackwash.tracker import IouTracker


def encode_jpeg(frame: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate(frame: np.ndarray, config: Config, hands, active_washer, gesture,
             collecting: bool = False) -> np.ndarray:
    import cv2  # noqa: PLC0415

    out = frame.copy()
    for rz in config.rack_zones:
        color = (0, 165, 255) if rz.requires_clear else (255, 0, 0)
        cv2.rectangle(out, (rz.x1, rz.y1), (rz.x2, rz.y2), color, 2)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    if active_washer:
        banner, color = f"Session: {active_washer}", (0, 255, 0)
    else:
        banner = "Session: none - show 1 (You) / 2 (Wife), show again to end"
        color = (0, 165, 255)
    cv2.putText(out, banner, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(out, f"gesture: {gesture}", (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if collecting:
        cv2.putText(out, "counting rack...", (8, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
        dish_detector=None,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._config = config
        self._store = store
        self._state = state
        self._clock = clock
        self._encode = jpeg_encoder
        self._annotate = annotator
        self._dish_detector = dish_detector
        self._running = False

        t = config.thresholds
        self._hand_tracker = IouTracker(
            iou_threshold=t.iou_match, coast_seconds=t.track_coast
        )
        self._session = SessionController(hold_seconds=t.gesture_hold)
        self._rack = RackDeltaCounter(
            config.rack_zones, set(config.dish_classes),
            rack_window=t.rack_window, dish_interval=t.dish_interval,
        )

    def _resolve_gesture(self, hands) -> str:
        for hand in hands:
            g = recognize_gesture(hand)
            if g in ("one", "two"):
                return g
        return "other"

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame), now)
        gesture = self._resolve_gesture(hands)
        active = self._session.update(gesture, now)
        detect = (
            (lambda: self._dish_detector.detect(frame))
            if self._dish_detector is not None
            else None
        )
        events = self._rack.process(active, hands, now, detect)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(
            frame, self._config, hands, active, gesture, self._rack.is_collecting
        )
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
