"""Dish detection behind a Protocol. The Ultralytics YOLO-World implementation
is the only file that imports ultralytics; swapping detectors is a one-file
change. Counting is dish-driven; identity comes from hands elsewhere."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from dishcounter.domain import Dish

Detection = tuple[tuple[int, int, int, int], str, float]  # bbox, label, conf


@runtime_checkable
class DishDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Dish]: ...


class FakeDishDetector:
    """Deterministic detector for tests/headless runs."""

    def __init__(self, script: list[list[Dish]]) -> None:
        if not script:
            script = [[]]
        self._script = script
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Dish]:
        dishes = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return dishes


def dishes_from_detections(detections: list[Detection], min_conf: float) -> list[Dish]:
    """Pure mapping from raw (bbox, label, conf) tuples to Dish objects."""
    return [
        Dish(id=None, bbox=bbox, label=label, confidence=conf)
        for bbox, label, conf in detections
        if conf >= min_conf
    ]


class YoloWorldDishDetector:
    """Real detector using Ultralytics open-vocabulary YOLO-World."""

    def __init__(  # pragma: no cover - loads model weights
        self,
        classes: list[str],
        conf: float,
        model: str = "yolov8s-worldv2.pt",
    ) -> None:
        from ultralytics import YOLOWorld  # noqa: PLC0415

        self._model = YOLOWorld(model)
        self._model.set_classes(classes)
        self._conf = conf

    def detect(self, frame: np.ndarray) -> list[Dish]:  # pragma: no cover - model I/O
        results = self._model.predict(frame, conf=self._conf, verbose=False)
        detections: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append(((x1, y1, x2, y2), label, conf))
        return dishes_from_detections(detections, self._conf)
