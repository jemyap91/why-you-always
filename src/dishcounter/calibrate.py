"""One-time (re-runnable) calibration: capture each person's median skin chroma
and let the user drag the sink rectangle. Persists only color profiles + zone
coords — never raw images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dishcounter.config import Config, SkinProfile, Thresholds, Zone
from dishcounter.domain import median_chroma


def profile_from_region(pixels_bgr: np.ndarray) -> SkinProfile:
    cr, cb = median_chroma(pixels_bgr)
    return SkinProfile(cr=cr, cb=cb)


def zone_from_drag(start: tuple[int, int], end: tuple[int, int]) -> Zone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return Zone(x1=x1, y1=y1, x2=x2, y2=y2)


def _sample_center_region(frame: np.ndarray, size: int | None = None) -> np.ndarray:
    h, w = frame.shape[:2]
    r = (min(h, w) // 6) if size is None else size // 2
    cx, cy = w // 2, h // 2
    crop = frame[cy - r: cy + r, cx - r: cx + r]
    return crop.reshape(-1, 3)


def run_calibration(
    config_path: str | Path = "config.yaml", camera_index: int = 0
) -> None:  # pragma: no cover - interactive, exercised manually
    import cv2  # noqa: PLC0415

    from dishcounter.camera import Camera

    cam = Camera(camera_index)
    profiles: dict[str, SkinProfile] = {}
    print("Calibration: hold each hand in the green box, press 'y' (You), 'w' (Wife).")
    while len(profiles) < 2:
        frame = cam.read()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        display = frame.copy()
        r = min(h, w) // 6
        cv2.rectangle(display, (w // 2 - r, h // 2 - r), (w // 2 + r, h // 2 + r),
                      (0, 255, 0), 2)
        cv2.imshow("calibrate - skin", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y"):
            profiles["You"] = profile_from_region(_sample_center_region(frame))
        elif key == ord("w"):
            profiles["Wife"] = profile_from_region(_sample_center_region(frame))
        elif key == 27:
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    frame = None
    while frame is None:
        frame = cam.read()
    roi = cv2.selectROI("drag the sink zone", frame, showCrosshair=True)
    x, y, bw, bh = (int(v) for v in roi)
    sink_zone = zone_from_drag((x, y), (x + bw, y + bh))
    cv2.destroyAllWindows()
    cam.release()

    config = Config(
        camera_index=camera_index,
        sink_zone=sink_zone,
        you_profile=profiles["You"],
        wife_profile=profiles["Wife"],
        thresholds=Thresholds(),
    )
    config.save(config_path)
    print(f"Saved calibration to {config_path}.")
