import numpy as np

from dishcounter.calibrate import (
    MIN_PROFILE_SEPARATION,
    profile_from_region,
    profile_separation,
    profiles_distinct,
    zone_from_drag,
)
from dishcounter.config import SkinProfile


def test_profiles_distinct_rejects_near_identical_profiles():
    # Regression for the real-world bug: two captures 0.3 apart can never be
    # told apart, so the guard must flag them.
    a = SkinProfile(cr=143.7, cb=115.3)
    b = SkinProfile(cr=143.9, cb=115.1)
    assert profile_separation(a, b) < MIN_PROFILE_SEPARATION
    assert profiles_distinct(a, b) is False


def test_profiles_distinct_accepts_well_separated_profiles():
    a = SkinProfile(cr=165.0, cb=110.0)
    b = SkinProfile(cr=120.0, cb=150.0)
    assert profiles_distinct(a, b) is True


def test_profile_from_region_matches_median_chroma():
    pixels = np.tile(np.array([80, 80, 220], dtype=np.uint8), (12, 1))
    profile = profile_from_region(pixels)
    assert profile.cr > 128  # reddish -> elevated Cr


def test_zone_from_drag_normalizes_corners():
    zone = zone_from_drag((120, 90), (20, 10))  # dragged up-left
    assert zone.x1 == 20
    assert zone.y1 == 10
    assert zone.x2 == 120
    assert zone.y2 == 90
    assert zone.contains((70, 50))


def test_calibration_module_does_not_reference_drying_zone():
    import inspect

    from dishcounter import calibrate

    source = inspect.getsource(calibrate)
    assert "drying" not in source
