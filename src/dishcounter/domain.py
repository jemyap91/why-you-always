"""Shared domain types. These are the ONLY types that cross component
boundaries — downstream components never see MediaPipe/OpenCV types."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class WashEvent:
    """A single counted (or uncertain) dish wash."""

    person: str  # "You" | "Wife" | "uncertain"
    timestamp: float
    confidence: float
    source_id: int  # tracker id of the dish (or hand) that produced the event
