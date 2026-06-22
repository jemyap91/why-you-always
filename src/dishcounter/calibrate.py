"""One-time (re-runnable) calibration: capture each person's median skin chroma
and let the user drag the sink rectangle. Persists only color profiles + zone
coords — never raw images."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from dishcounter.config import Config, SkinProfile, Thresholds, Zone
from dishcounter.domain import median_chroma
from dishcounter.identity import AMBIGUOUS_MARGIN

# The classifier refuses to commit unless one profile is at least AMBIGUOUS_MARGIN
# closer than the other, so two profiles must be comfortably farther apart than
# that to ever be told apart in practice.
MIN_PROFILE_SEPARATION = 2 * AMBIGUOUS_MARGIN


def profile_from_region(pixels_bgr: np.ndarray) -> SkinProfile:
    cr, cb = median_chroma(pixels_bgr)
    return SkinProfile(cr=cr, cb=cb)


def profile_separation(a: SkinProfile, b: SkinProfile) -> float:
    """YCrCb distance between two skin profiles."""
    return math.dist((a.cr, a.cb), (b.cr, b.cb))


def profiles_distinct(a: SkinProfile, b: SkinProfile) -> bool:
    """True if the two profiles are far enough apart to be distinguishable."""
    return profile_separation(a, b) >= MIN_PROFILE_SEPARATION


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
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("Calibration:")
    print("  - Fill the green box with ONE hand, then press 'y' (You) / 'w' (Wife).")
    print("  - 'r' resets,  ENTER saves once both are captured,  Esc cancels.")

    while True:
        frame = cam.read()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        r = min(h, w) // 6
        cx, cy = w // 2, h // 2
        display = frame.copy()
        cv2.rectangle(display, (cx - r, cy - r), (cx + r, cy + r), (0, 255, 0), 2)

        # Live readout of what would be captured right now.
        live_cr, live_cb = median_chroma(_sample_center_region(frame))
        cv2.putText(display, f"box now: cr{live_cr:.0f} cb{live_cb:.0f}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)

        y0 = 62
        for name, k in (("You", "y"), ("Wife", "w")):
            if name in profiles:
                p = profiles[name]
                txt = f"{name}: captured  cr{p.cr:.0f} cb{p.cb:.0f}"
                col = (0, 255, 0)
            else:
                txt = f"{name}: press '{k}'"
                col = (0, 165, 255)
            cv2.putText(display, txt, (10, y0), font, 0.7, col, 2)
            y0 += 30

        if len(profiles) == 2:
            sep = profile_separation(profiles["You"], profiles["Wife"])
            if sep < MIN_PROFILE_SEPARATION:
                cv2.putText(display, f"TOO SIMILAR (gap {sep:.1f}) - press 'r' to redo",
                            (10, y0), font, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, f"OK (gap {sep:.1f}) - press ENTER to save",
                            (10, y0), font, 0.7, (0, 255, 0), 2)

        cv2.imshow("calibrate - skin", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):
            profiles["You"] = profile_from_region(_sample_center_region(frame))
            print(f"Captured You:  cr={profiles['You'].cr:.1f} cb={profiles['You'].cb:.1f}")
        elif key == ord("w"):
            profiles["Wife"] = profile_from_region(_sample_center_region(frame))
            print(f"Captured Wife: cr={profiles['Wife'].cr:.1f} cb={profiles['Wife'].cb:.1f}")
        elif key == ord("r"):
            profiles.clear()
            print("Reset - recapture both hands.")
        elif key in (13, 10) and len(profiles) == 2:  # Enter / Return
            if not profiles_distinct(profiles["You"], profiles["Wife"]):
                sep = profile_separation(profiles["You"], profiles["Wife"])
                print(
                    f"WARNING: profiles are only {sep:.1f} apart (need "
                    f">= {MIN_PROFILE_SEPARATION:.0f}); washes will likely stay "
                    "'uncertain'. Press 'r' to recapture, or ENTER again to save anyway."
                )
                # Consume one more frame so a held ENTER doesn't instantly re-trigger.
                if (cv2.waitKey(0) & 0xFF) not in (13, 10):
                    continue
            break
        elif key == 27:  # Esc
            cv2.destroyAllWindows()
            cam.release()
            print("Calibration cancelled.")
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
