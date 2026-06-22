import numpy as np
import pytest

from dishcounter.config import Config, SkinProfile, Thresholds, Zone


@pytest.fixture
def sample_config() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        drying_zone=Zone(x1=100, y1=0, x2=200, y2=100),
        you_profile=SkinProfile(cr=165.0, cb=110.0),
        wife_profile=SkinProfile(cr=120.0, cb=150.0),
        thresholds=Thresholds(identity_distance=60.0),
    )


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((120, 200, 3), dtype=np.uint8)
