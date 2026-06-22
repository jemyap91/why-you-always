import numpy as np

from dishcounter.calibrate import profile_from_region, zone_from_drag


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
