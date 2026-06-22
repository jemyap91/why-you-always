from dishcounter.cli import build_parser, main


def test_parser_has_calibrate_and_run_subcommands():
    parser = build_parser()
    cal = parser.parse_args(["calibrate", "--camera", "2"])
    assert cal.command == "calibrate"
    assert cal.camera == 2
    runargs = parser.parse_args(["run", "--port", "9000"])
    assert runargs.command == "run"
    assert runargs.port == 9000


def test_run_without_config_exits_with_calibrate_hint(tmp_path, capsys):
    missing = tmp_path / "config.yaml"
    code = main(["run", "--config", str(missing)])
    assert code == 2
    assert "calibrate" in capsys.readouterr().err


def test_no_command_prints_help_and_returns_nonzero(capsys):
    assert main([]) == 1
