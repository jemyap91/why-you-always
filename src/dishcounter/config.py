"""Pydantic-backed configuration persisted to YAML. Written by `calibrate`,
read by `run`. Holds zones and tuning thresholds."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


DEFAULT_DISH_CLASSES = [
    "plate",
    "bowl",
    "cup",
    "glass",
    "mug",
    "fork",
    "knife",
    "spoon",
]


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
    exit_grace: float = 1.5          # seconds a dish must be gone before counting
    cooldown: float = 3.0            # seconds before the same person can re-fire
    iou_match: float = 0.3           # IoU needed to keep a track's id
    gesture_hold: float = 1.0        # seconds a gesture must be held to act


class Config(BaseModel):
    camera_index: int
    sink_zone: Zone
    dish_classes: list[str] = DEFAULT_DISH_CLASSES
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"
    thresholds: Thresholds = Thresholds()

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
