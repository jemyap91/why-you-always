import numpy as np
import pytest

from dishcounter.config import Config, Thresholds, Zone


@pytest.fixture
def sample_config() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        thresholds=Thresholds(gesture_hold=1.0),
    )


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((120, 200, 3), dtype=np.uint8)
