# Dish Counter

A local computer-vision scoreboard that keeps a fair, hands-free tally of how many
dishes **You** vs **Wife** wash at a shared kitchen sink, watched by a webcam.

- **A wash is counted** when your hand dwells in the **sink zone** for a few seconds,
  **a dish is seen in the sink during that dwell**, and then your hand **leaves**.
  The dish check (a webcam object detector) is what stops hand-rinsing or
  sponge-wringing from being counted — no dish in the sink, no count.
- **Attribution is by hand gesture** — show **1 finger** to start a *You* session,
  **2 fingers** to start a *Wife* session; **show the same number again to end** it
  (or the other number to switch). Every wash while a session is active is credited
  to that person. With no active session nothing is counted — safe. No face
  recognition, no stored images.
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

This downloads OpenCV and MediaPipe and installs a `dishcounter` command inside
`.venv`. Confirm it worked:

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

> Tip: make the sink zone snug around the basin, not the whole counter. Leaving the
> sink zone is what triggers the count, so a tight boundary gives you the cleanest
> signal.

---

## 4. Run

```bash
.venv/bin/dishcounter run
```

On first run, MediaPipe downloads its hand landmark model (~8 MB). Subsequent runs
start immediately.

The dish detector (YOLO-World) also downloads its weights on first run
(~340MB, including the CLIP text encoder). It runs only while a session is active
and a hand is in the sink, so it stays idle when nobody is washing.

Then open **http://127.0.0.1:8000** in a browser. You'll see:

- The **live camera feed** with:
  - The **sink zone** drawn in blue.
  - A bounding box around each detected **hand** — **green** when the hand is inside
    the sink zone, **amber** otherwise.
  - A **session banner** at the top of the frame (`Session: You` / `Session: Wife` /
    `Session: none - show 1 (You) / 2 (Wife), show again to end`) and a
    `gesture: <name>` label below it.
- A **scoreboard**: *You* vs *Wife*, showing **today** and **all-time** totals.
- A **"camera offline"** notice if the webcam disconnects (the app keeps retrying).

**Workflow:**

1. Hold **1 finger** up toward the camera for about 1 second — **away from the sink
   basin** (gestures are only read outside the sink zone, so your washing hand can't
   change the session by accident). The banner switches to `Session: You`.
2. Wash a dish. Keep your hand in the sink for a few seconds (the default is 3 s),
   then lift the dish out and carry it away. The hand leaving the sink zone triggers
   the count.
3. Repeat for each dish. When you're done, hold **1 finger** again for about 1
   second — the session ends and counting stops until the next gesture.
4. The next person holds **2 fingers** to start a *Wife* session, then **2 fingers**
   again to end it (or just show 1/2 to switch directly).

With no active session nothing is counted — so handing off or walking away is safe.

Counts are saved to **`dishcounter.db`** (a local SQLite file) and persist across
restarts. Leave the command running while you do dishes; stop it with `Ctrl-C`.

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

`config.yaml` holds a `thresholds` block you can edit by hand.

**`thresholds` block:**

| Setting | Default | What it does | When to change |
|---|---|---|---|
| `min_wash` | `3.0` | Seconds a hand must dwell inside the sink zone before the exit is counted as a wash | **Raise** if brief hand-passes get counted; **lower** if real washes are missed because you're quick |
| `cooldown` | `3.0` | Minimum seconds before the same **person** can be counted again | **Raise** if a single wash is being double-counted |
| `track_coast` | `2.0` | Seconds a lost hand track is kept alive so brief detector dropouts (suds, occlusion) don't break the dwell timer | **Raise** if the hand tracker loses the hand mid-wash and resets the dwell; **lower** if two washes in quick succession get merged into one |
| `gesture_hold` | `1.0` | Seconds a gesture must be held before a session starts or ends | **Raise** if sessions start too easily (accidental gestures); **lower** for snappier switching |
| `iou_match` | `0.3` | How much a hand box must overlap frame-to-frame to be treated as the same hand | Rarely needs changing |
| `dish_interval` | `0.5` | Minimum seconds between dish-detector runs during a wash | Lower for more frequent dish checks (more CPU); raise to save CPU |
| `dish_min_hits` | `1` | Dish-in-sink confirmations needed to count a wash; `0` disables the dish gate | Raise if you still see false counts; set `0` to count on hand activity alone |

After editing, just restart `dishcounter run`.

---

## 7. Development

The entire pipeline is testable **without a camera or any model weights** — a fake
camera and a fake hand detector drive the whole flow, so the suite runs anywhere
(MediaPipe is never imported in tests):

```bash
.venv/bin/pytest         # full suite, no hardware needed
.venv/bin/ruff check .   # lint
```

### How it works

```
Vision thread:  Camera ─ HandDetector (MediaPipe) ─ IouTracker ─ recognize_gesture ─ SessionController ─┐
                                                                                                         ├─ WashCycleEngine ─ CountStore
                                                                                                         │  (publishes frame + counts)
Web thread:     FastAPI → streams the latest annotated frame + scoreboard to the browser
```

**The hand leaving the sink zone drives the count; the active session supplies
identity.** A single MediaPipe hand detector feeds an IoU tracker that assigns stable
ids across frames (with coasting to bridge brief detection dropouts). `recognize_gesture`
reads the hand landmarks each frame (one/two/fist/other). `SessionController` debounces
those results — a gesture must be held for `gesture_hold` seconds before it acts — and
exposes an `active` washer (`"You"`, `"Wife"`, or `None`). `WashCycleEngine` tracks how
long each hand id has been continuously inside the sink zone; when a hand leaves (or
disappears from frame), it fires a `WashEvent` if the dwell reached `min_wash` seconds
and an active session is set. A short per-person `cooldown` collapses bursts (e.g. a
two-handed carry). With no active session the engine records nothing.

The vision thread never blocks on the browser, so the frame rate is independent of how
many tabs are open. Counts are **derived from an append-only event log**, so the totals
are always reconstructable and a single bad frame can never corrupt the score.

The hand detector sits behind a small `HandDetector` interface, so swapping models is a
one-file change.

See `docs/superpowers/specs/` for the design spec and `docs/superpowers/plans/` for the
implementation plan.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run` exits asking you to calibrate | No `config.yaml` yet — run `dishcounter calibrate`. |
| Dashboard shows "camera offline" | Webcam unplugged or wrong index — reconnect, or try `--camera 1`. |
| `pip install` fails on MediaPipe | The venv isn't Python 3.12 — recreate it with the `brew --prefix python@3.12` interpreter (Step 2). |
| Washes aren't being counted | Make sure the sink zone is snug around the basin (re-run `calibrate` if needed), and keep your hand in the sink long enough for the dwell timer — watch for the hand box turning green to confirm the zone is right. |
| Washes counted too early or mid-wash | **Raise** `min_wash` so the hand must stay in the sink longer before the exit triggers a count. |
| Overcounting due to hand flickering | **Raise** `track_coast` so brief detection dropouts don't reset the dwell timer and produce extra counts. |
| Session won't start | Hold 1 or 2 fingers steady, facing the camera, for the full `gesture_hold` duration (~1 s). Watch the `gesture:` label in the feed to confirm the hand is being read correctly. |
| Session starts accidentally | Raise `gesture_hold` in `config.yaml` so a longer deliberate hold is required. |
| Wrong person's session is active | Show the correct finger count (1 or 2) to switch directly, or repeat the active number to end the session. |
| One wash counts twice | Raise `cooldown` in `config.yaml`. |
| Washes not counted during a valid session | Check the session banner shows the right name. Also check the sink zone is drawn around the basin — re-run `calibrate` if needed. |

---

## Privacy

The app stores only:

- `config.yaml` — the sink rectangle and threshold settings. No skin profiles, no
  biometric data of any kind.
- `dishcounter.db` — rows of `(person, timestamp, confidence, counted)`.

No video, no photos of faces or hands, are ever written to disk.
