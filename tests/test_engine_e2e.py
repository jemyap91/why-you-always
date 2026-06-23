from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.dish_detector import FakeDishDetector
from dishcounter.domain import Dish, Hand
from dishcounter.engine import Engine
from dishcounter.state import SharedState
from dishcounter.store import CountStore

_FINGERS = {"index": (8, 6, 0.40), "middle": (12, 10, 0.48),
            "ring": (16, 14, 0.56), "pinky": (20, 18, 0.64)}


def _landmarks(extended):
    pts = [(0.5, 0.9)] * 21
    for name, (tip, pip, x) in _FINGERS.items():
        pts[pip] = (x, 0.5)
        pts[tip] = (x, 0.25 if name in extended else 0.55)
    return pts


def _hand(cx, cy, extended):
    return Hand(id=None, bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
                landmarks=_landmarks(extended))


def _fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def _engine(cfg, hand_script, frames, times, dish_detector=None):
    return Engine(
        FakeCamera(frames),
        FakeHandDetector(hand_script),
        cfg,
        CountStore(":memory:"),
        SharedState(),
        clock=_fake_clock(times),
        jpeg_encoder=lambda frame: b"jpeg",
        dish_detector=dish_detector,
    )


def test_gesture_session_then_wash_counts_one_for_you(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})                          # gesture, outside sink
    wash = _hand(50, 50, {"index", "middle", "ring"})         # 'other', in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
    )
    engine.run()  # FakeCamera exhausts after 5 frames
    assert engine._store.totals(8.0)["all_time"] == {"You": 1, "Wife": 0}


def test_wash_without_a_session_counts_nothing(sample_config, blank_frame):
    wash = _hand(50, 50, {"index", "middle", "ring"})
    engine = _engine(
        sample_config,
        hand_script=[[wash], [wash], []],
        frames=[blank_frame] * 3,
        times=[0.0, 3.5, 6.0],
    )
    engine.run()
    assert engine._store.totals(6.0)["all_time"] == {"You": 0, "Wife": 0}


def test_gesture_from_hand_in_sink_is_ignored(sample_config):
    # Regression: the washing hand (inside the sink zone) must not drive the
    # session — only a hand held OUTSIDE the sink counts as a deliberate signal.
    engine = _engine(sample_config, hand_script=[[]], frames=[], times=[])
    in_sink_one = _hand(50, 50, {"index"})      # 'one' but centroid in sink (0..100)
    assert engine._resolve_gesture([in_sink_one]) == "other"
    out_one = _hand(150, 150, {"index"})         # 'one' outside the sink
    assert engine._resolve_gesture([out_one]) == "one"


_DISH_IN_SINK = Dish(id=None, bbox=(40, 40, 60, 60), label="plate", confidence=0.9)


def test_wash_with_dish_in_sink_counts_one(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})                          # gesture, outside sink
    wash = _hand(50, 50, {"index", "middle", "ring"})         # 'other', in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
        dish_detector=FakeDishDetector([[_DISH_IN_SINK]]),  # always sees a plate
    )
    engine.run()
    assert engine._store.totals(8.0)["all_time"] == {"You": 1, "Wife": 0}


def test_same_wash_motion_without_a_dish_counts_zero(sample_config, blank_frame):
    # Core regression: identical hand motion, but the dish detector sees nothing.
    one = _hand(150, 150, {"index"})
    wash = _hand(50, 50, {"index", "middle", "ring"})
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], []],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.5, 4.6, 8.0],
        dish_detector=FakeDishDetector([[]]),  # never sees a dish
    )
    engine.run()
    assert engine._store.totals(8.0)["all_time"] == {"You": 0, "Wife": 0}


def test_dish_detector_runs_only_in_sink_and_throttled(sample_config, blank_frame):
    class _Spy:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            return [_DISH_IN_SINK]

    spy = _Spy()
    one = _hand(150, 150, {"index"})                   # outside sink (gesture)
    wash = _hand(50, 50, {"index", "middle", "ring"})  # in sink
    engine = _engine(
        sample_config,
        hand_script=[[one], [one], [wash], [wash], [wash]],
        frames=[blank_frame] * 5,
        times=[0.0, 1.0, 1.2, 1.4, 2.0],
        dish_detector=spy,
    )
    for now in [0.0, 1.0, 1.2, 1.4, 2.0]:
        engine.process_frame(blank_frame, now)
    # Not called on the two gesture frames (no hand in sink). In-sink frames at
    # 1.2 (first), 1.4 (throttled out: <0.5s later), 2.0 (>=0.5s later) -> 2 runs.
    assert spy.calls == 2
