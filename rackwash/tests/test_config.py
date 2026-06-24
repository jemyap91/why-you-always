import pytest

from rackwash.config import Config, RackZone, Thresholds, Zone


def test_zone_contains_and_overlaps():
    z = Zone(x1=0, y1=0, x2=100, y2=100)
    assert z.contains((50, 50)) is True
    assert z.contains((150, 50)) is False
    assert z.overlaps((50, 50, 150, 150)) is True   # corner overlap
    assert z.overlaps((101, 0, 200, 100)) is False   # just right of zone
    assert z.overlaps((100, 100, 200, 200)) is True  # touching corner counts


def test_rack_zone_defaults_requires_clear_true():
    rz = RackZone(x1=0, y1=0, x2=10, y2=10)
    assert rz.requires_clear is True
    assert rz.contains((5, 5)) is True


def test_thresholds_defaults():
    t = Thresholds()
    assert t.gesture_hold == 1.5
    assert t.rack_window == 1.5
    assert t.dish_interval == 0.5
    assert t.iou_match == 0.3
    assert t.track_coast == 2.0


def test_config_round_trips_rack_zones(tmp_path):
    path = tmp_path / "config.yaml"
    Config(
        camera_index=0,
        rack_zones=[
            RackZone(x1=0, y1=0, x2=100, y2=100, requires_clear=False),
            RackZone(x1=120, y1=0, x2=200, y2=80, requires_clear=True),
        ],
    ).save(path)
    loaded = Config.load(path)
    assert len(loaded.rack_zones) == 2
    assert loaded.rack_zones[0].requires_clear is False
    assert loaded.rack_zones[1].requires_clear is True
    assert "plate" in loaded.dish_classes and "fork" not in loaded.dish_classes


def test_load_missing_file_tells_user_to_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Config.load(tmp_path / "nope.yaml")
