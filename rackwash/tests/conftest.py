import numpy as np
import pytest

from rackwash.config import Config, RackZone, Thresholds


@pytest.fixture
def sample_config() -> Config:
    return Config(
        camera_index=0,
        rack_zones=[RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False)],
        thresholds=Thresholds(
            gesture_hold=1.5,
            rack_window=1.0,
            dish_interval=0.4,
            # coast=0: compressed test timeline; real sessions span minutes, far
            # exceeding the 2s default coast, so the gesture hand expires and the
            # session resets between gestures. Zeroing it models that here.
            track_coast=0.0,
        ),
    )


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((120, 200, 3), dtype=np.uint8)
