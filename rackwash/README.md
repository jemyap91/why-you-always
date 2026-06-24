# Rackwash

A local computer-vision scoreboard that tallies how many dishes **You** vs
**Wife** wash — by counting the **net increase of dishware in your drying racks**
over a washing session, not by watching hands. Outcome-based: a count only
happens when a clean dish actually appears in a rack.

Standalone package — it does not depend on `dishcounter`.

## Requirements

- macOS (or Linux) with a webcam.
- Python 3.12 (MediaPipe has no 3.14 support): `brew install python@3.12`.

## Setup

Installs into the shared project venv (reuses the heavy MediaPipe/YOLO wheels):

```bash
.venv/bin/python -m pip install -e ./rackwash
```

## Calibrate

```bash
.venv/bin/rackwash calibrate
```

For each drying rack, drag a rectangle and press **Enter**, then press **`b`** if
your body blocks that rack while washing, or **`a`** if it stays visible. Press
**Esc** (without dragging) when you've added all your racks.

Then drag the **sign-in box** — a small box in a corner, **away from the racks
and your normal hand paths**. Gestures (1/2 fingers) are only read inside this
box, so incidental hand poses while washing can't flip the session. Press
**Esc** to skip it (gestures then register from any hand — the old, flip-prone
behavior). Saves `config.yaml` (rectangles only — no images).

## Run

```bash
.venv/bin/rackwash run        # open http://127.0.0.1:8000
```

Start a session by holding **1 finger** (You) or **2** (Wife) for ~1.5s **inside
the sign-in box** (shown in yellow on the feed); show the same number again to
end it, or the other number to switch. The app counts the
dishware added to your racks during the session and credits it to that washer.
On first run, MediaPipe and YOLO-World download their weights (~hundreds of MB).
YOLO runs only in short bursts at the start and end of each session.

## How it works

```
Camera -> MediaPipe(hands) -> IouTracker -> recognize_gesture -> SessionController
       -> RackDeltaCounter (YOLO bursts at boundaries) -> CountStore
```

At each session boundary the `RackDeltaCounter` runs a short YOLO burst over the
rack regions, takes a robust per-rack dishware count (median over clean frames,
skipping person/hand-occluded frames for racks you marked as body-blocked), and
credits `max(0, end - start)` per rack to the session's washer. Counts persist in
`rackwash.db`.

## Develop

```bash
.venv/bin/python -m pytest rackwash/tests -q
.venv/bin/ruff check rackwash
```

## Privacy

Only the rack rectangles (`config.yaml`) and event rows (`rackwash.db`) are
stored. No images are ever persisted.
