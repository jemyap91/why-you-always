from rackwash.camera import FakeCamera
from rackwash.detector import FakeHandDetector
from rackwash.domain import Dish, Hand
from rackwash.engine import Engine
from rackwash.state import SharedState
from rackwash.store import CountStore

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


def _plate(cx, cy=50):
    return Dish(id=None, bbox=(cx - 5, cy - 5, cx + 5, cy + 5), label="plate", confidence=0.9)


class _RackFake:
    def __init__(self):
        self.dishes = []

    def detect(self, frame):
        return list(self.dishes)


def _engine(cfg, hand_script, rack):
    return Engine(
        FakeCamera([]),
        FakeHandDetector(hand_script),
        cfg,
        CountStore(":memory:"),
        SharedState(),
        clock=lambda: 0.0,
        jpeg_encoder=lambda frame: b"jpeg",
        dish_detector=rack,
    )


def test_gesture_registers_anywhere(sample_config):
    engine = _engine(sample_config, [[]], _RackFake())
    assert engine._resolve_gesture([_hand(50, 50, {"index"})]) == "one"


def test_full_session_counts_rack_delta_for_you(sample_config, blank_frame):
    one = _hand(150, 150, {"index"})
    none = []
    rack = _RackFake()
    script = [[one], [one], none, none, [one], [one], none, none]
    engine = _engine(sample_config, script, rack)

    engine.process_frame(blank_frame, 0.0)   # one (hold begins)
    engine.process_frame(blank_frame, 1.5)   # one held 1.5 -> You; start burst sample 0
    engine.process_frame(blank_frame, 1.9)   # burst sample 0
    engine.process_frame(blank_frame, 2.5)   # finalize start burst (baseline 0)

    rack.dishes = [_plate(20), _plate(40)]

    engine.process_frame(blank_frame, 5.0)   # one (hold begins)
    engine.process_frame(blank_frame, 6.5)   # one held 1.5 -> session ends; end burst sample 2
    engine.process_frame(blank_frame, 6.9)   # burst sample 2
    engine.process_frame(blank_frame, 7.5)   # finalize -> delta 2

    assert engine._store.totals(7.5)["all_time"] == {"You": 2, "Wife": 0}
