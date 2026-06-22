import numpy as np

from dishcounter.camera import FakeCamera
from dishcounter.detector import FakeHandDetector
from dishcounter.domain import Hand
from dishcounter.engine import Engine
from dishcounter.state import SharedState
from dishcounter.store import CountStore

RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (16, 1))   # -> You
BLUE = np.tile(np.array([220, 90, 90], dtype=np.uint8), (16, 1))  # -> Wife


def _hand(cx, region):
    # Width-40 box so consecutive frames overlap (IoU tracking keeps the id),
    # modelling a hand moving continuously across the sink->drying boundary.
    return Hand(id=None, bbox=(cx - 20, 45, cx + 20, 55), region_pixels=region)


def _fake_clock(times):
    it = iter(times)
    return lambda: next(it)


def test_full_pipeline_counts_one_wash_for_you(sample_config, blank_frame):
    # Frame 0: hand centroid at cx=95 (inside sink zone 0..100).
    # Frame 1: hand centroid at cx=105 (inside drying zone 100..200).
    # The width-40 boxes overlap (IoU ~0.6) so IoU tracking keeps the same id.
    script = [[_hand(95, RED)], [_hand(105, RED)]]
    detector = FakeHandDetector(script)
    camera = FakeCamera([blank_frame, blank_frame])
    store = CountStore(":memory:")
    state = SharedState()
    engine = Engine(
        camera,
        detector,
        sample_config,
        store,
        state,
        clock=_fake_clock([100.0, 101.0]),
        jpeg_encoder=lambda frame: b"jpeg",
    )

    engine.run()  # FakeCamera exhausts after 2 frames

    now = 101.0
    assert store.totals(now)["all_time"] == {"You": 1, "Wife": 0}
    snap = state.snapshot()
    assert snap["camera_online"] is True
    assert snap["frame_jpeg"] == b"jpeg"
    assert snap["counts"]["all_time"]["You"] == 1


def test_process_frame_publishes_counts(sample_config, blank_frame):
    detector = FakeHandDetector([[_hand(95, BLUE)], [_hand(105, BLUE)]])
    engine = Engine(
        FakeCamera([]),
        detector,
        sample_config,
        CountStore(":memory:"),
        SharedState(),
        clock=lambda: 0.0,
        jpeg_encoder=lambda frame: b"x",
    )
    engine.process_frame(blank_frame, now=0.0)
    events = engine.process_frame(blank_frame, now=1.0)
    assert len(events) == 1
    assert events[0].person == "Wife"
