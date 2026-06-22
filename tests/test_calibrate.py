from dishcounter.calibrate import zone_from_drag


def test_zone_from_drag_normalizes_corners():
    zone = zone_from_drag((120, 90), (20, 10))  # dragged up-left
    assert zone.x1 == 20
    assert zone.y1 == 10
    assert zone.x2 == 120
    assert zone.y2 == 90
    assert zone.contains((70, 50))


def test_calibration_module_does_not_reference_skin_or_drying():
    import inspect

    from dishcounter import calibrate

    src = inspect.getsource(calibrate)
    assert "drying" not in src
    assert "SkinProfile" not in src
    assert "profile" not in src
