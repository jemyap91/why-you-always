import numpy as np

from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.dish_detector import FakeDishDetector
from dishcounter.domain import Dish, Hand
from dishcounter.engine import Engine
from dishcounter.state import SharedState
from dishcounter.store import CountStore

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))   # -> You
BLUE = np.tile(np.array([220, 90, 90], dtype=np.uint8), (16, 1))  # -> Wife


def _hand(cx, region):
    return Hand(id=None, bbox=(cx - 20, 40, cx + 20, 60), region_pixels=region)


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


def test_full_pipeline_counts_one_wash_for_you(sample_config, blank_frame):
    # Frame 0: hand + dish in the sink (cx=50, sink zone 0..100) -> lock You.
    # Frame 1: both gone; 2.0s > exit_grace (1.5) -> fire one You wash.
    engine = _engine(
        sample_config,
        hand_script=[[_hand(50, RED)], []],
        dish_script=[[_dish(50)], []],
        frames=[blank_frame, blank_frame],
        times=[100.0, 102.0],
    )
    engine.run()  # FakeCamera exhausts after 2 frames

    assert engine_store_totals(engine) == {"You": 1, "Wife": 0}


def engine_store_totals(engine):
    return engine._store.totals(10_000_000_000.0)["all_time"]


def test_process_frame_publishes_counts(sample_config, blank_frame):
    engine = _engine(
        sample_config,
        hand_script=[[_hand(50, BLUE)], []],
        dish_script=[[_dish(50)], []],
        frames=[],
        times=[],
    )
    engine.process_frame(blank_frame, now=0.0)        # dish in sink -> lock Wife
    events = engine.process_frame(blank_frame, now=2.0)  # dish gone -> fire
    assert len(events) == 1
    assert events[0].person == "Wife"
