"""Command-line entry points: `dishcounter calibrate` and `dishcounter run`."""

from __future__ import annotations

import argparse
import sys
import threading

from dishcounter.config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dishcounter")
    sub = parser.add_subparsers(dest="command")

    cal = sub.add_parser("calibrate", help="capture skin profiles and draw zones")
    cal.add_argument("--config", default="config.yaml")
    cal.add_argument("--camera", type=int, default=0)

    run = sub.add_parser("run", help="start the counter + dashboard")
    run.add_argument("--config", default="config.yaml")
    run.add_argument("--db", default="dishcounter.db")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "calibrate":
        from dishcounter.calibrate import run_calibration  # noqa: PLC0415

        run_calibration(args.config, args.camera)
        return 0

    if args.command == "run":
        try:
            config = Config.load(args.config)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        _serve(config, args.db, args.host, args.port)
        return 0

    parser.print_help(sys.stderr)
    return 1


def _serve(config: Config, db: str, host: str, port: int) -> None:  # pragma: no cover
    from dishcounter.camera import Camera  # noqa: PLC0415
    from dishcounter.dashboard import run_server  # noqa: PLC0415
    from dishcounter.detector import MediaPipeHandDetector  # noqa: PLC0415
    from dishcounter.engine import Engine  # noqa: PLC0415
    from dishcounter.state import SharedState  # noqa: PLC0415
    from dishcounter.store import CountStore  # noqa: PLC0415

    from dishcounter.dish_detector import YoloWorldDishDetector  # noqa: PLC0415

    state = SharedState()
    engine = Engine(
        Camera(config.camera_index),
        MediaPipeHandDetector(),
        YoloWorldDishDetector(config.dish_classes, config.dish_conf, config.yolo_model),
        config,
        CountStore(db),
        state,
    )
    threading.Thread(target=engine.run, daemon=True).start()
    print(f"Dashboard at http://{host}:{port}")
    run_server(state, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
