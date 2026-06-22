"""Shared domain types. These are the ONLY types that cross component
boundaries — downstream components never see MediaPipe/OpenCV types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Hand:
    """A detected hand for one frame."""

    id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Dish:
    """A detected dish for one frame, from the object detector."""

    id: int | None
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    label: str = ""
    confidence: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class WashEvent:
    """A single counted (or uncertain) dish wash."""

    person: str  # "You" | "Wife" | "uncertain"
    timestamp: float
    confidence: float
    source_id: int  # tracker id of the dish (or hand) that produced the event


def bgr_to_ycrcb(b: float, g: float, r: float) -> tuple[float, float, float]:
    """BT.601 conversion matching cv2.COLOR_BGR2YCrCb. Inputs/outputs 0-255."""
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return (y, cr, cb)


def median_chroma(pixels_bgr: np.ndarray) -> tuple[float, float]:
    """Median (Cr, Cb) over an (N, 3) BGR pixel array. Median is robust to
    the suds/occlusion outliers that washing introduces."""
    arr = np.asarray(pixels_bgr, dtype=np.float64).reshape(-1, 3)
    b, g, r = arr[:, 0], arr[:, 1], arr[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128.0
    cb = (b - y) * 0.564 + 128.0
    return (float(np.median(cr)), float(np.median(cb)))
