from rackwash.calibrate import rack_zone_from_drag


def test_rack_zone_from_drag_normalizes_and_sets_flag():
    rz = rack_zone_from_drag((120, 90), (20, 10), requires_clear=True)
    assert (rz.x1, rz.y1, rz.x2, rz.y2) == (20, 10, 120, 90)
    assert rz.requires_clear is True

    rz2 = rack_zone_from_drag((0, 0), (50, 40), requires_clear=False)
    assert rz2.requires_clear is False
    assert rz2.contains((25, 20)) is True
