from rackwash.calibrate import rack_zone_from_drag


def test_rack_zone_from_drag_normalizes_and_sets_flag():
    rz = rack_zone_from_drag((120, 90), (20, 10), requires_clear=True)
    assert (rz.x1, rz.y1, rz.x2, rz.y2) == (20, 10, 120, 90)
    assert rz.requires_clear is True

    rz2 = rack_zone_from_drag((0, 0), (50, 40), requires_clear=False)
    assert rz2.requires_clear is False
    assert rz2.contains((25, 20)) is True


def test_sign_in_zone_from_drag_normalizes_corners():
    from rackwash.calibrate import sign_in_zone_from_drag

    z = sign_in_zone_from_drag((200, 50), (150, 0))  # dragged bottom-right to top-left
    assert (z.x1, z.y1, z.x2, z.y2) == (150, 0, 200, 50)
    assert z.contains((175, 25))
