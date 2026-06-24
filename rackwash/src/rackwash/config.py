"""Pydantic config persisted to YAML. Written by `calibrate`, read by `run`.
Holds the drying-rack regions and tuning thresholds."""

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

    def overlaps(self, bbox: tuple[int, int, int, int]) -> bool:
        bx1, by1, bx2, by2 = bbox
        lo_x, hi_x = sorted((self.x1, self.x2))
        lo_y, hi_y = sorted((self.y1, self.y2))
        blo_x, bhi_x = sorted((bx1, bx2))
        blo_y, bhi_y = sorted((by1, by2))
        return not (bhi_x < lo_x or blo_x > hi_x or bhi_y < lo_y or blo_y > hi_y)


class RackZone(Zone):
    requires_clear: bool = True  # skip counting on frames where a person/hand overlaps it


class Thresholds(BaseModel):
    iou_match: float = 0.3       # IoU needed to keep a hand track's id
    gesture_hold: float = 1.5    # seconds a gesture must be held to act
    track_coast: float = 2.0     # seconds a lost hand track is kept alive
    rack_window: float = 1.5     # seconds a boundary counting burst lasts
    dish_interval: float = 0.5   # min seconds between dish-detector runs in a burst


class Config(BaseModel):
    camera_index: int
    rack_zones: list[RackZone] = Field(default_factory=list)
    sign_in_zone: Zone | None = None  # gestures only count inside this box (if set)
    thresholds: Thresholds = Thresholds()
    dish_classes: list[str] = Field(
        default_factory=lambda: ["plate", "bowl", "cup", "glass", "mug"]
    )
    dish_conf: float = 0.4
    yolo_model: str = "yolov8s-worldv2.pt"

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No config at {path}. Run `rackwash calibrate` first."
            )
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
