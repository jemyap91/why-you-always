"""One-time (re-runnable) calibration: drag a rectangle around each drying rack
and mark whether your body blocks it while washing. Persists only the rack
rectangles — never raw images. Identity is by gesture."""

from __future__ import annotations

from pathlib import Path

from rackwash.config import Config, RackZone, Thresholds, Zone


def rack_zone_from_drag(
    start: tuple[int, int], end: tuple[int, int], requires_clear: bool
) -> RackZone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return RackZone(x1=x1, y1=y1, x2=x2, y2=y2, requires_clear=requires_clear)


def sign_in_zone_from_drag(start: tuple[int, int], end: tuple[int, int]) -> Zone:
    x1, x2 = sorted((start[0], end[0]))
    y1, y2 = sorted((start[1], end[1]))
    return Zone(x1=x1, y1=y1, x2=x2, y2=y2)


def run_calibration(
    config_path: str | Path = "config.yaml", camera_index: int = 0
) -> None:  # pragma: no cover - interactive, exercised manually
    import cv2  # noqa: PLC0415

    from rackwash.camera import Camera

    cam = Camera(camera_index)
    print(
        "Calibration: for each drying rack, drag a rectangle and press ENTER.\n"
        "Then press 'b' if your body blocks that rack while washing, else 'a'.\n"
        "Press ESC (no drag) when you have added all racks."
    )
    rack_zones: list[RackZone] = []
    while True:
        frame = None
        while frame is None:
            frame = cam.read()
        roi = cv2.selectROI("drag a rack zone (ESC to finish)", frame, showCrosshair=True)
        x, y, bw, bh = (int(v) for v in roi)
        if bw == 0 or bh == 0:
            break
        key = -1
        while key not in (ord("a"), ord("b")):
            key = cv2.waitKey(0) & 0xFF
        rack_zones.append(rack_zone_from_drag((x, y), (x + bw, y + bh), key == ord("b")))
    cv2.destroyAllWindows()

    if not rack_zones:
        cam.release()
        print("No rack zones drawn; nothing saved.")
        return

    print(
        "Now drag the SIGN-IN box — a corner away from the racks and your hands.\n"
        "Gestures (1/2 fingers) only count inside this box. ESC to skip."
    )
    frame = None
    while frame is None:
        frame = cam.read()
    roi = cv2.selectROI("drag the sign-in box (ESC to skip)", frame, showCrosshair=True)
    x, y, bw, bh = (int(v) for v in roi)
    sign_in_zone = (
        sign_in_zone_from_drag((x, y), (x + bw, y + bh)) if bw and bh else None
    )
    cv2.destroyAllWindows()
    cam.release()
    if sign_in_zone is None:
        print("No sign-in box drawn; gestures will register from any hand.")

    Config(
        camera_index=camera_index, rack_zones=rack_zones,
        sign_in_zone=sign_in_zone, thresholds=Thresholds()
    ).save(config_path)
    print(f"Saved {len(rack_zones)} rack zone(s) to {config_path}.")
