import numpy as np

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


def _hand(extended):
    return Hand(id=None, bbox=(150, 150, 190, 190), landmarks=_landmarks(extended))


def _dish(cx):
    return Dish(id=None, bbox=(cx - 20, 40, cx + 20, 60), label="plate", confidence=0.9)


def _fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def _engine(sample_config, hand_script, dish_script, frames, times):
    return Engine(
        FakeCamera(frames),
        FakeHandDetector(hand_script),
        FakeDishDetector(dish_script),
        sample_config,
        CountStore(":memory:"),
        SharedState(),
        clock=_fake_clock(times),
        jpeg_encoder=lambda frame: b"jpeg",
    )


def test_session_started_with_one_finger_counts_dish_for_you(sample_config, blank_frame):
    one = _hand({"index"})
    engine = _engine(
        sample_config,
        # hold 'one' across two frames to clear the 1.0s gesture_hold, then idle.
        hand_script=[[one], [one], [one], []],
        dish_script=[[], [], [_dish(50)], []],   # dish appears in sink once session active
        frames=[blank_frame] * 4,
        times=[0.0, 1.0, 1.2, 3.5],
    )
    engine.run()  # FakeCamera exhausts after 4 frames
    assert engine._store.totals(3.5)["all_time"] == {"You": 1, "Wife": 0}


def test_dish_washed_with_no_session_is_not_counted(sample_config, blank_frame):
    engine = _engine(
        sample_config,
        hand_script=[[], []],
        dish_script=[[_dish(50)], []],
        frames=[blank_frame, blank_frame],
        times=[0.0, 2.0],
    )
    engine.run()
    assert engine._store.totals(2.0)["all_time"] == {"You": 0, "Wife": 0}
