# Dish Counter

A local computer-vision scoreboard that keeps a fair, hands-free tally of how many
dishes **You** vs **Wife** wash at a shared kitchen sink, watched by a webcam.

- **Attribution is by hand skin tone** — no face recognition, no stored images.
- **A wash is counted** when a tracked hand moves from the **sink zone** into the
  **drying zone** (it counts the hand-off, not the dish).
- **Biased toward undercounting** — when the app isn't confident *whose* hand it is,
  it records the event as `uncertain` and leaves it **out** of the score. It would
  rather miss a dish than credit the wrong person.

Everything runs on your machine. The only things ever saved are two small skin-tone
color profiles and a log of count events — never any pictures of faces or hands.

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

This downloads OpenCV and MediaPipe (a few hundred MB the first time) and installs a
`dishcounter` command inside `.venv`. Confirm it worked:

```bash
.venv/bin/dishcounter --help
```

You should see the `calibrate` and `run` subcommands.

---

## 3. Calibrate (one time, re-runnable)

The app needs to learn each person's skin tone and where the sink and drying areas
are in the camera view. Run:

```bash
.venv/bin/dishcounter calibrate
```

A camera window opens. Follow these steps:

1. **Capture skin tones.** A green box appears in the center of the frame.
   - Have the **first person** hold their hand so it fills the green box, then press
     **`y`** (for *You*).
   - Have the **second person** do the same, then press **`w`** (for *Wife*).
   - Press **`Esc`** at any time to cancel.
2. **Draw the sink zone.** A new window titled *"drag the sink zone"* appears.
   Click-drag a rectangle around the **sink** area, then press **`Enter`** (or
   `Space`) to confirm.
3. **Draw the drying zone.** Repeat for the **drying rack / counter** area where a
   clean dish ends up, and press **`Enter`**.

This writes **`config.yaml`** to the repo root containing only the two color profiles
and the two zone rectangles — no images. Re-run `calibrate` any time the camera moves
or the lighting changes a lot.

> Tip: calibrate under the **same lighting** you'll actually wash dishes in. Skin-tone
> matching uses color, so a big lighting change can lower confidence (which makes the
> app count *less*, never wrong).

---

## 4. Run

```bash
.venv/bin/dishcounter run
```

Then open **http://127.0.0.1:8000** in a browser. You'll see:

- The **live camera feed** with the sink zone (blue) and drying zone (green) drawn on
  it, plus a box around each detected hand.
- A **scoreboard**: *You* vs *Wife*, showing **today** and **all-time** totals.
- A **"camera offline"** notice if the webcam disconnects (the app keeps retrying).

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

`config.yaml` has a `thresholds` block you can edit by hand. Defaults:

| Setting | Default | What it does | When to change |
|---|---|---|---|
| `identity_distance` | `25.0` | How close a hand's color must be to a saved profile to be attributed | **Raise** if too many washes come out `uncertain`; **lower** if people are being confused for each other |
| `presence_window` | `5.0` | Seconds a sink visit stays "fresh" before a crossing counts | Raise if washes take longer than ~5s between sink and drying |
| `cooldown` | `3.0` | Minimum seconds before the same hand can count again | **Raise** if a single wash is being double-counted |
| `iou_match` | `0.3` | How much a hand box must overlap frame-to-frame to be treated as the same hand | Rarely needs changing |

After editing, just restart `dishcounter run`. If attribution feels off, the most
reliable fix is to **re-run `calibrate`** under your real lighting.

---

## 7. Development

The entire pipeline is testable **without a camera** — fake camera and fake detector
implementations drive the whole flow, so the suite runs anywhere:

```bash
.venv/bin/pytest         # 54 tests, no hardware needed
.venv/bin/ruff check .   # lint
```

### How it works

```
Vision thread:  Camera → HandDetector → HandTracker → IdentityClassifier
                       → ZoneEventEngine → CountStore   (publishes frame + counts)
Web thread:     FastAPI → streams the latest annotated frame + scoreboard to the browser
```

The vision thread never blocks on the browser, so the frame rate is independent of how
many tabs are open. Counts are **derived from an append-only event log**, so the totals
are always reconstructable and a single bad frame can never corrupt the score.

The hand detector sits behind a small `HandDetector` interface, so swapping MediaPipe
for another model is a one-file change.

See `docs/superpowers/specs/` for the design spec and `docs/superpowers/plans/` for the
implementation plan.

---

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run` exits asking you to calibrate | No `config.yaml` yet — run `dishcounter calibrate`. |
| Dashboard shows "camera offline" | Webcam unplugged or wrong index — reconnect, or try `--camera 1`. |
| `pip install` fails on MediaPipe | The venv isn't Python 3.12 — recreate it with the `brew --prefix python@3.12` interpreter (Step 2). |
| Lots of washes are `uncertain` (not scored) | Lighting differs from calibration — re-`calibrate`, or raise `identity_distance`. |
| One wash counts twice | Raise `cooldown` in `config.yaml`. |
| Wrong person credited | Lower `identity_distance`, or re-`calibrate` so the two profiles are more distinct. |

---

## Privacy

The app stores only:

- `config.yaml` — two skin-tone color values (chroma) and two rectangles.
- `dishcounter.db` — rows of `(person, timestamp, confidence, counted)`.

No video, no photos of faces or hands, are ever written to disk.
