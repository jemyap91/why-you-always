"""Pydantic-backed configuration persisted to YAML. Written by `calibrate`,
read by `run`. Holds zones and tuning thresholds."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Zone(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        lo_x, hi_x = sorted((self.x1, self.x2))
        lo_y, hi_y = sorted((self.y1, self.y2))
        return lo_x <= x <= hi_x and lo_y <= y <= hi_y


class Thresholds(BaseModel):
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act
    track_coast: float = 2.0         # seconds a lost hand track is kept alive
    min_wash: float = 3.0            # seconds a hand must dwell in the sink to count
    dish_interval: float = 0.5       # min seconds between dish-detector runs
    dish_min_hits: int = 1           # dish-in-sink confirmations to validate a wash


class Config(BaseModel):
    camera_index: int
    sink_zone: Zone
    thresholds: Thresholds = Thresholds()
    dish_classes: list[str] = Field(
        default_factory=lambda: [
            "plate", "bowl", "cup", "glass", "mug", "fork", "knife", "spoon",
        ]
    )
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No config at {path}. Run `dishcounter calibrate` first."
            )
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
