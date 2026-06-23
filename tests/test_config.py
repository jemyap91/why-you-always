import pytest

from dishcounter.config import Config, Thresholds, Zone


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
    assert t.cooldown == 3.0
    assert t.iou_match == 0.3


def _sample_config() -> Config:
    return Config(
        camera_index=0,
        sink_zone=Zone(x1=0, y1=0, x2=100, y2=100),
        thresholds=Thresholds(),
    )


def test_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    _sample_config().save(path)
    loaded = Config.load(path)
    assert loaded.sink_zone.contains((50, 50))
    assert loaded.thresholds.cooldown == 3.0


def test_config_load_missing_file_tells_user_to_calibrate(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        Config.load(tmp_path / "nope.yaml")


def test_config_has_no_skin_profiles_and_no_identity_distance():
    from dishcounter.config import Config, Zone

    cfg = Config(camera_index=0, sink_zone=Zone(x1=0, y1=0, x2=10, y2=10))
    assert not hasattr(cfg, "you_profile")
    assert not hasattr(cfg, "wife_profile")
    assert not hasattr(cfg.thresholds, "identity_distance")
    assert cfg.thresholds.gesture_hold == 1.0


def test_dish_fields_have_defaults():
    from dishcounter.config import Thresholds

    t = Thresholds()
    assert t.dish_interval == 0.5
    assert t.dish_min_hits == 1


def test_config_without_dish_fields_still_loads(tmp_path):
    # An old config (sink zone + a couple thresholds, no dish fields) must load.
    import yaml

    from dishcounter.config import Config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "camera_index": 0,
                "sink_zone": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                "thresholds": {"min_wash": 3.0, "cooldown": 3.0},
            }
        )
    )
    cfg = Config.load(path)
    assert cfg.dish_conf == 0.4
    assert cfg.yolo_model == "yolov8s-worldv2.pt"
    assert "plate" in cfg.dish_classes
    assert cfg.thresholds.dish_min_hits == 1
