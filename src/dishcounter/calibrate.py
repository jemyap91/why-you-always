"""One-time (re-runnable) calibration: drag the sink rectangle. Persists only
the sink zone — never raw images. Identity is by gesture, not skin tone."""

from __future__ import annotations

from pathlib import Path

from dishcounter.config import Config, Thresholds, Zone


def zone_from_drag(start: tuple[int, int], end: tuple[int, int]) -> Zone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return Zone(x1=x1, y1=y1, x2=x2, y2=y2)


def run_calibration(
    config_path: str | Path = "config.yaml", camera_index: int = 0
) -> None:  # pragma: no cover - interactive, exercised manually
    import cv2  # noqa: PLC0415

    from dishcounter.camera import Camera

    cam = Camera(camera_index)
    print("Calibration: drag a rectangle around the sink, then press ENTER.")
    frame = None
    while frame is None:
        frame = cam.read()
    roi = cv2.selectROI("drag the sink zone", frame, showCrosshair=True)
    x, y, bw, bh = (int(v) for v in roi)
    sink_zone = zone_from_drag((x, y), (x + bw, y + bh))
    cv2.destroyAllWindows()
    cam.release()

    config = Config(
        camera_index=camera_index, sink_zone=sink_zone, thresholds=Thresholds()
    )
    config.save(config_path)
    print(f"Saved calibration to {config_path}.")
