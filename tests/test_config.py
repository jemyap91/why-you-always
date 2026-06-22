import pytest

from dishcounter.config import Config, SkinProfile, Thresholds, Zone


def test_zone_contains_inclusive_edges():
    z = Zone(x1=0, y1=0, x2=10, y2=10)
    assert z.contains((0, 0))
    assert z.contains((10, 10))
    assert z.contains((5, 5))
    assert not z.contains((11, 5))


def test_zone_normalizes_inverted_corners():
    z = Zone(x1=10, y1=10, x2=0, y2=0)  # dragged bottom-right to top-left
    assert z.contains((5, 5))


def test_thresholds_have_spec_defaults():
    t = Thresholds()
    assert t.identity_distance == 25.0
    assert t.exit_grace == 1.5
    assert t.cooldown == 3.0
    assert t.iou_match == 0.3


def _sample_config() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        you_profile=SkinProfile(cr=150.0, cb=110.0),
        wife_profile=SkinProfile(cr=140.0, cb=120.0),
        thresholds=Thresholds(),
    )


def test_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    _sample_config().save(path)
    loaded = Config.load(path)
    assert loaded.you_profile.cr == 150.0
    assert loaded.sink_zone.contains((50, 50))
    assert loaded.thresholds.cooldown == 3.0


def test_config_load_missing_file_tells_user_to_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Config.load(tmp_path / "nope.yaml")


def test_config_has_dish_settings_and_exit_grace():
    cfg = Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        you_profile=SkinProfile(cr=165.0, cb=110.0),
        wife_profile=SkinProfile(cr=120.0, cb=150.0),
    )
    assert "plate" in cfg.dish_classes
    assert cfg.dish_conf == 0.4
    assert cfg.yolo_model == "yolov8s-worldv2.pt"
    assert cfg.thresholds.exit_grace == 1.5
