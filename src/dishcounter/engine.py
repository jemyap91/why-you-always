"""Wires the vision pipeline and owns the capture loop. Dependency-injected
(camera, detector, clock, jpeg encoder) so the whole pipeline runs in tests
with zero hardware."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from dishcounter.config import Config
from dishcounter.dish_detector import DishDetector
from dishcounter.domain import WashEvent, median_chroma
from dishcounter.fusion import FusionEngine
from dishcounter.identity import IdentityClassifier
from dishcounter.ring import ring_metrics
from dishcounter.state import SharedState
from dishcounter.store import CountStore
from dishcounter.tracker import IouTracker


def encode_jpeg(frame: np.ndarray) -> bytes:
    import cv2  # noqa: PLC0415

    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def annotate(frame: np.ndarray, config: Config, hands, dishes, identity=None
             ) -> np.ndarray:
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
        label, conf = identity.classify(hand) if identity else ("hand", 0.0)
        identified = label in ("You", "Wife")
        color = (0, 255, 0) if identified else (0, 165, 255)  # green vs amber
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        if hand.region_pixels is not None and len(hand.region_pixels):
            cr, cb = median_chroma(hand.region_pixels)
            text += f"  cr{cr:.0f} cb{cb:.0f}"
        cv2.putText(out, text, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        rm = ring_metrics(frame, hand)
        if rm is not None:
            mpct, rcr = rm
            cv2.putText(out, f"ring metal{mpct:.0f}% cr{rcr:.0f}",
                        (x1, min(out.shape[0] - 4, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    hud = f"hands:{len(hands)}  dishes:{len(dishes)}"
    cv2.putText(out, hud, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
        self._identity = IdentityClassifier(
            config.you_profile, config.wife_profile, max_distance=t.identity_distance
        )
        self._fusion = FusionEngine(
            config.sink_zone,
            self._identity,
            exit_grace=t.exit_grace,
            cooldown=t.cooldown,
        )

    def process_frame(self, frame: np.ndarray, now: float) -> list[WashEvent]:
        hands = self._hand_tracker.update(self._detector.detect(frame))
        dishes = self._dish_tracker.update(self._dish_detector.detect(frame))
        events = self._fusion.process(hands, dishes, now)
        for event in events:
            self._store.record(event)
        annotated = self._annotate(frame, self._config, hands, dishes, self._identity)
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
