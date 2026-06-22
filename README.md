# Dish Counter

A local computer-vision scoreboard that keeps a fair, hands-free tally of how many
dishes **You** vs **Wife** wash at a shared kitchen sink, watched by a webcam.

- **Dishes are detected directly** — an open-vocabulary object detector (YOLO-World)
  finds plates, bowls, cups, and cutlery in the frame.
- **A wash is counted** when a dish that was washed at the **sink** then leaves the
  camera's view (you carry it off to put it away). There's no drying rack to watch —
  *out of view = washed*.
- **Attribution is by hand gesture** — show **1 finger** to start a *You* session,
  **2 fingers** to start a *Wife* session, or a **fist** to end the session. Every
  dish washed while a session is active is credited to that person. With no active
  session nothing is counted — safe. No face recognition, no stored images.
- **Biased toward undercounting** — when no session is active, the app records
  nothing rather than guessing who washed a dish.

Everything runs on your machine. The only things ever saved are the sink rectangle,
tuning settings, and a log of count events — never any pictures of faces or hands.

---

## 1. Requirements

- **macOS** (or Linux) with a connected **webcam** (built for a Logitech Brio, works
  with any OpenCV-compatible camera).
- **Python 3.12** specifically. MediaPipe (the hand detector) does not yet support
  Python 3.13/3.14, so the project pins 3.12:

  ```bash
  brew install python@3.12
  ```

---

## 2. Setup (one time)

From the repo root, create a virtual environment **using the 3.12 interpreter** and
install the package:

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

This downloads OpenCV, MediaPipe, and Ultralytics (a few hundred MB the first time) and
installs a `dishcounter` command inside `.venv`. Confirm it worked:

```bash
.venv/bin/dishcounter --help
```

You should see the `calibrate` and `run` subcommands.

---

## 3. Calibrate (one time, re-runnable)

The app needs to know where the sink is in the camera view. Run:

```bash
.venv/bin/dishcounter calibrate
```

A camera window opens. **Draw the sink zone** — click-drag a rectangle around the
**sink** area (where washing happens), then press **`Enter`** (or `Space`) to confirm.

This writes **`config.yaml`** to the repo root containing only the sink rectangle and
tuning settings — no images, no skin profiles. Re-run `calibrate` any time the camera
moves.

> Tip: make the sink zone snug around the basin, not the whole counter. Only dishes
> that pass through this zone get counted.

---

## 4. Run

```bash
.venv/bin/dishcounter run
```

On **first run**, YOLO-World downloads its weights plus a CLIP text-encoder for the
open-vocabulary prompts (**~340 MB total**), and Ultralytics may auto-install a couple
of small extras (`clip`, `ftfy`) — you'll see a one-time "restart runtime" notice that
you can ignore. Subsequent runs start immediately. The dishes the detector looks for
are configurable in `config.yaml` (`dish_classes`).

Then open **http://127.0.0.1:8000** in a browser. You'll see:

- The **live camera feed** with:
  - The **sink zone** drawn in blue.
  - A labelled green box around each detected **dish**.
  - A **session banner** at the top of the frame (`Session: You` / `Session: Wife` /
    `Session: none - show 1 finger (You) / 2 (Wife)`) and a `gesture: <name>` label
    below it.
- A **scoreboard**: *You* vs *Wife*, showing **today** and **all-time** totals.
- A **"camera offline"** notice if the webcam disconnects (the app keeps retrying).

**Workflow:**

1. Hold **1 finger** up toward the camera for about 1 second — the banner switches to
   `Session: You`.
2. Wash your dishes. Each dish that leaves view after being in the sink zone is counted
   to *You*.
3. Hold a **fist** for about 1 second when you're done — the session ends and counting
   stops until the next gesture.
4. The next person holds **2 fingers** to start a *Wife* session, then a **fist** to
   end it.

With no active session nothing is counted — so handing off or walking away is safe.

Counts are saved to **`dishcounter.db`** (a local SQLite file) and persist across
restarts. Leave the command running while you do dishes; stop it with `Ctrl-C`.

> Running two detectors (dishes + hands) per frame is heavier than hand-only — on a
> typical laptop CPU expect a low single-digit frame rate. That's fine: counting is
> event-based on a dish leaving view, not frame-rate sensitive.

If you run `dishcounter run` before calibrating, it stops with a clear message telling
you to run `calibrate` first.

---

## 5. Command reference

```
dishcounter calibrate [--config config.yaml] [--camera 0]
dishcounter run       [--config config.yaml] [--db dishcounter.db]
                      [--host 127.0.0.1] [--port 8000]
```

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config.yaml` | Where calibration is read/written |
| `--camera` | `0` | Webcam index (try `1`, `2`… if you have several) |
| `--db` | `dishcounter.db` | SQLite file for the event log |
| `--host` | `127.0.0.1` | Dashboard bind address |
| `--port` | `8000` | Dashboard port |

---

## 6. Tuning (optional)

`config.yaml` holds the dish-detection settings and a `thresholds` block you can edit by
hand.

**Dish detection (top level of `config.yaml`):**

| Setting | Default | What it does | When to change |
|---|---|---|---|
| `dish_classes` | `[plate, bowl, cup, glass, mug, fork, knife, spoon]` | The text prompts YOLO-World looks for | Add/remove items to match what you actually wash |
| `dish_conf` | `0.4` | Minimum detector confidence to accept a dish | **Lower** if real dishes are missed; **raise** if random objects get detected |
| `yolo_model` | `yolov8s-worldv2.pt` | YOLO-World weights | Use a larger variant (`…m`/`…l`) for accuracy at the cost of speed |

**`thresholds` block:**

| Setting | Default | What it does | When to change |
|---|---|---|---|
| `gesture_hold` | `1.0` | Seconds a gesture must be held before a session starts or ends | **Raise** if sessions start too easily (accidental gestures); **lower** for snappier switching |
| `exit_grace` | `1.5` | Seconds a dish must be gone from view before it's counted (debounces detection flicker) | **Raise** if a dish flickers out mid-wash and gets counted early; **lower** for a snappier count |
| `cooldown` | `3.0` | Minimum seconds before the same **person** can be counted again | **Raise** if a single wash is being double-counted |
| `iou_match` | `0.3` | How much a box must overlap frame-to-frame to be treated as the same hand/dish | Rarely needs changing |

After editing, just restart `dishcounter run`.

---

## 7. Development

The entire pipeline is testable **without a camera or any model weights** — fake
camera, fake hand detector, and fake dish detector implementations drive the whole
flow, so the suite runs anywhere (Ultralytics/MediaPipe are never imported in tests):

```bash
.venv/bin/pytest         # 60 tests, no hardware needed
.venv/bin/ruff check .   # lint
```

### How it works

```
Vision thread:  Camera ─┬─ HandDetector (MediaPipe) ─ IouTracker ─ recognize_gesture ─ SessionController ─┐
                        │                                                                                   ├─ FusionEngine ─ CountStore
                        └─ DishDetector (YOLO-World) ─ IouTracker ─────────────────────────────────────────┘   (publishes frame + counts)
Web thread:     FastAPI → streams the latest annotated frame + scoreboard to the browser
```

**Dishes drive the count; the active session supplies identity.** Each detector is
tracked separately (stable ids across frames via IoU). `recognize_gesture` reads the
hand landmarks each frame (one/two/fist/other). `SessionController` debounces those
results — a gesture must be held for `gesture_hold` seconds before it acts — and
exposes an `active` washer (`"You"`, `"Wife"`, or `None`). The `FusionEngine` uses the
active washer as the identity for any dish that exits the sink zone, then fires one
`WashEvent` when that dish disappears from view for longer than `exit_grace`. A short
per-person `cooldown` collapses bursts (e.g. a two-handed carry). With no active
session the engine records nothing.

The vision thread never blocks on the browser, so the frame rate is independent of how
many tabs are open. Counts are **derived from an append-only event log**, so the totals
are always reconstructable and a single bad frame can never corrupt the score.

Both detectors sit behind small `HandDetector` / `DishDetector` interfaces, so swapping
a model is a one-file change.

See `docs/superpowers/specs/` for the design spec and `docs/superpowers/plans/` for the
implementation plan.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run` exits asking you to calibrate | No `config.yaml` yet — run `dishcounter calibrate`. |
| Dashboard shows "camera offline" | Webcam unplugged or wrong index — reconnect, or try `--camera 1`. |
| `pip install` fails on MediaPipe | The venv isn't Python 3.12 — recreate it with the `brew --prefix python@3.12` interpreter (Step 2). |
| Dishes aren't being detected | **Lower** `dish_conf`, or add the right item to `dish_classes`. Make sure the dish is clearly visible at the sink. |
| Washes counted while still washing | **Raise** `exit_grace` so brief detector dropouts during scrubbing aren't read as the dish leaving. |
| Session won't start | Hold 1 or 2 fingers steady, facing the camera, for the full `gesture_hold` duration (~1 s). Watch the `gesture:` label in the feed to confirm the hand is being read correctly. |
| Session starts accidentally | Raise `gesture_hold` in `config.yaml` so a longer deliberate hold is required. |
| Wrong person's session is active | Show a **fist** to end the current session, then show the correct finger count to start a new one. |
| One wash counts twice | Raise `cooldown` in `config.yaml`. |
| Dishes not counted during a valid session | Check the session banner is green and shows the right name. Also check the sink zone is drawn around the basin — re-run `calibrate` if needed. |

---

## Privacy

The app stores only:

- `config.yaml` — the sink rectangle and dish-detection/threshold settings. No skin
  profiles, no biometric data of any kind.
- `dishcounter.db` — rows of `(person, timestamp, confidence, counted)`.

No video, no photos of faces or hands, are ever written to disk.
