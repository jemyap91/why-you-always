import yaml

from rackwash.cli import build_parser, main


def test_parser_has_calibrate_and_run():
    p = build_parser()
    assert p.parse_args(["calibrate", "--camera", "2"]).camera == 2
    assert p.parse_args(["run", "--port", "9000"]).port == 9000


def test_run_missing_config_exits_2_with_calibrate_hint(tmp_path, capsys):
    code = main(["run", "--config", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "calibrate" in capsys.readouterr().err


def test_run_empty_rack_zones_exits_2_with_rack_hint(tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"camera_index": 0, "rack_zones": []}))
    code = main(["run", "--config", str(path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "calibrate" in err and "rack" in err


def test_no_command_returns_1(capsys):
    assert main([]) == 1
