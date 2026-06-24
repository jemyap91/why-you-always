"""Command-line entry points: `rackwash calibrate` and `rackwash run`."""

from __future__ import annotations

import argparse
import sys
import threading

from rackwash.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rackwash")
    sub = parser.add_subparsers(dest="command")

    cal = sub.add_parser("calibrate", help="draw your drying-rack regions")
    cal.add_argument("--config", default="config.yaml")
    cal.add_argument("--camera", type=int, default=0)

    run = sub.add_parser("run", help="start the counter + dashboard")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--db", default="rackwash.db")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        from rackwash.calibrate import run_calibration  # noqa: PLC0415

        run_calibration(args.config, args.camera)
        return 0

    if args.command == "run":
        try:
            config = Config.load(args.config)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if not config.rack_zones:
            print(
                "No rack zones configured. Run `rackwash calibrate` first to draw "
                "your rack region(s).",
                file=sys.stderr,
            )
            return 2
        _serve(config, args.db, args.host, args.port)
        return 0

    parser.print_help(sys.stderr)
    return 1


def _serve(config: Config, db: str, host: str, port: int) -> None:  # pragma: no cover
    from rackwash.camera import Camera  # noqa: PLC0415
    from rackwash.dashboard import run_server  # noqa: PLC0415
    from rackwash.detector import MediaPipeHandDetector  # noqa: PLC0415
    from rackwash.dish_detector import YoloWorldDishDetector  # noqa: PLC0415
    from rackwash.engine import Engine  # noqa: PLC0415
    from rackwash.state import SharedState  # noqa: PLC0415
    from rackwash.store import CountStore  # noqa: PLC0415

    state = SharedState()
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        config,
        CountStore(db),
        state,
        dish_detector=YoloWorldDishDetector(
            [*config.dish_classes, "person"], config.dish_conf, config.yolo_model
        ),
    )
    threading.Thread(target=engine.run, daemon=True).start()
    print(f"Dashboard at http://{host}:{port}")
    run_server(state, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
