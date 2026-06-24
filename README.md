# Rackwash

A local computer-vision scoreboard that keeps a fair, hands-free tally of how many
dishes **You** vs **Wife** wash, watched by a webcam. It counts by **results in the
drying rack**, not by watching hands at the sink.

- **Outcome-based counting.** A wash session is bracketed by a gesture (start/end).
  At each boundary the app runs a short object-detection burst over your drying-rack
  region(s) and counts the dishware sitting there. The **increase** across the
  session — `dishes_at_end − dishes_at_start` — is credited to that session's washer.
  Nothing is counted by watching hands move, so rinsing, sponge-wringing, and
  reaching never inflate the score.
- **Attribution is by hand gesture, inside a sign-in box.** Show **1 finger** (You)
  or **2 fingers** (Wife) **inside a small calibrated "sign-in" box** for ~1.5s to
  start; show the same number again to end, or the other number to switch. Gestures
  are only read inside that box, so incidental hand poses while washing can't flip
  the session. With no active session nothing is counted.
- **Private by design.** The only things saved are the rack/sign-in rectangles,
  tuning settings, and a log of count events — never any images of faces or hands.

> The repository also contains a legacy `dishcounter` package (an earlier
> sink-dwell approach under `src/dishcounter/`). **Rackwash** (`rackwash/`) is the
> current project; this README describes it.

---

## 1. Requirements

- **macOS** (or Linux) with a connected **webcam**.
- **Python 3.12** specifically — MediaPipe does not yet support 3.13/3.14:

  ```bash
  brew install python@3.12
  ```

---

## 2. Setup (one time)

From the repo root, create a 3.12 virtual environment and install the rackwash
package:

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "./rackwash[dev]"
```

Confirm it worked:

```bash
.venv/bin/rackwash --help        # shows: calibrate, run
```

---

## 3. Calibrate (one time, re-runnable)

```bash
.venv/bin/rackwash calibrate
```

A camera window opens. Two things to set up:

1. **Drying-rack zones.** For each rack, drag a rectangle and press **Enter**, then
   press **`b`** if your body blocks that rack while washing, or **`a`** if it stays
   visible. Press **Esc** (without dragging) when you've added all your racks.
2. **Sign-in box.** Drag one more box — a small box in a **corner, away from the
   racks and your normal hand paths**. Gestures are only read inside this box, so
   your washing/reaching hands can't change the session by accident. Press **Esc**
   to skip it (gestures then register from *any* hand — the old, flip-prone
   behavior).

This writes **`config.yaml`** (rectangles + tuning only — no images). Re-run any
time the camera moves.

> Tip: put the sign-in box somewhere you can reach with one hand without leaning
> over the sink, and where you don't normally wave your hands.

---

## 4. Run

```bash
.venv/bin/rackwash run        # then open http://127.0.0.1:8000
```

On first run, MediaPipe (hand landmarks) and YOLO-World (dishware) download their
weights (a few hundred MB total). YOLO runs only in **short bursts at session
boundaries**, so it's idle the rest of the time.

The dashboard shows:

- The **live feed** with rack zones (orange = body-blocked, blue = always-visible),
  the **sign-in box** (yellow), hand boxes, a **session banner**
  (`Session: You` / `Wife` / `none — show 1 (You) / 2 (Wife)`), and the current
  `gesture:` reading. A **zoom +/−/Reset** control magnifies the preview.
- A **scoreboard**: *You* vs *Wife*, **today** and **all-time**.

**Workflow:**

1. Hold **1 finger inside the sign-in box** for ~1.5s — the banner switches to
   `Session: You`.
2. Wash and load your racks as normal.
3. When you're done, hold **1 finger in the box** again to end — the app counts the
   dishware added to the racks during your session and credits it to You.
4. The next person holds **2 fingers in the box** to start a *Wife* session, then
   **2 fingers** again to end (or show 1/2 to switch directly).

Counts persist in **`rackwash.db`** across restarts. Stop with `Ctrl-C`.

---

## 5. Command reference

```
rackwash calibrate [--config config.yaml] [--camera 0]
rackwash run       [--config config.yaml] [--db rackwash.db]
                   [--host 127.0.0.1] [--port 8000]
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config.yaml` | Where calibration is read/written |
| `--camera` | `0` | Webcam index (try `1`, `2`… if you have several) |
| `--db` | `rackwash.db` | SQLite file for the event log |
| `--host` | `127.0.0.1` | Dashboard bind address |
| `--port` | `8000` | Dashboard port |

---

## 6. Tuning (optional)

`config.yaml` holds a `thresholds` block plus dish-detector settings.

**`thresholds`:**

| Setting | Default | What it does | When to change |
|---|---|---|---|
| `gesture_hold` | `1.5` | Seconds a gesture must be held (inside the sign-in box) before it acts | **Raise** if sessions start too easily; **lower** for snappier switching |
| `track_coast` | `2.0` | Seconds a lost hand track is kept alive to bridge detector dropouts | Rarely needs changing |
| `iou_match` | `0.3` | Frame-to-frame overlap to treat a hand as the same track | Rarely needs changing |
| `rack_window` | `1.5` | Seconds of the counting burst at each session boundary | Raise if a boundary count is unstable |
| `dish_interval` | `0.5` | Min seconds between dish-detector runs within a burst | Raise to save CPU; lower for more samples |

**Dish detection (top level):** `dish_classes` (what to count — e.g. plate, bowl,
cup, glass, mug), `dish_conf` (min confidence), `yolo_model` (weights).

After editing, restart `rackwash run`.

---

## 7. How it works

```
Camera → MediaPipe(hands) → IouTracker → recognize_gesture → SessionController ┐
                                                                               ├→ RackDeltaCounter → CountStore
                                          YOLO-World(dishware), boundary bursts ┘   (publishes frame + counts)
Web thread: FastAPI streams the latest annotated frame + scoreboard to the browser
```

A MediaPipe hand detector feeds an IoU tracker. `recognize_gesture` reads finger
poses, but the engine only accepts a gesture from a hand **inside the sign-in box**.
`SessionController` debounces it (held `gesture_hold` seconds) into an active washer
(`You` / `Wife` / `None`), toggling per number. At each session **boundary** the
`RackDeltaCounter` runs a short YOLO-World burst over the rack zones, counts the
dishware present, and credits `max(0, end − start)` per rack to the session's washer
(racks marked "body blocks it" are skipped on frames where a person/hand overlaps
them). Counts are an **append-only event log**, so totals are always reconstructable.

The vision thread never blocks on the browser. Detectors sit behind small
interfaces, so the whole pipeline runs in tests with fakes and no hardware.

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design notes.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run` says no rack zones / asks to calibrate | Run `rackwash calibrate` and draw at least one rack zone. |
| Dashboard shows "camera offline" | Webcam unplugged or wrong index — reconnect, or try `--camera 1`. |
| `pip install` fails on MediaPipe | The venv isn't Python 3.12 — recreate it with the `brew --prefix python@3.12` interpreter (Step 2). |
| Sessions keep switching / ending by themselves | Make sure you calibrated a **sign-in box** and only gesture inside it. If you skipped it, re-run `calibrate` and draw one. |
| Session won't start | Hold 1 or 2 fingers steady **inside the sign-in box** for the full `gesture_hold` (~1.5s). Watch the `gesture:` label to confirm it's read. |
| Counts seem low/high at a boundary | Make sure the rack zone tightly frames where dishes land, and that nothing blocks it during the end burst; tune `rack_window`. |
| Wrong person credited | Show the correct number in the sign-in box to switch, or repeat the active number to end. |

---

## 9. Development

```bash
.venv/bin/python -m pytest rackwash/tests   # rackwash suite, no hardware needed
.venv/bin/ruff check rackwash/src rackwash/tests
```

The whole pipeline is testable without a camera or model weights — fakes drive the
camera and detectors, and neither MediaPipe nor YOLO is imported in tests.

---

## Privacy

The app stores only:

- `config.yaml` — rack/sign-in rectangles and threshold settings. No biometric data.
- `rackwash.db` — rows of `(person, timestamp, confidence, counted)`.

No video, no photos of faces or hands, are ever written to disk.
