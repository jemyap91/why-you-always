from dishcounter.domain import Hand
from dishcounter.gesture import recognize_gesture

# Finger -> (tip_idx, pip_idx, x). Thumb is ignored.
_FINGERS = {"index": (8, 6, 0.40), "middle": (12, 10, 0.48),
            "ring": (16, 14, 0.56), "pinky": (20, 18, 0.64)}


def _landmarks(extended: set[str], pointing: str = "up") -> list[tuple[float, float]]:
    if pointing == "up":
        wrist, ext_y, curl_y, pip_y = (0.5, 0.9), 0.25, 0.55, 0.5
    else:
        wrist, ext_y, curl_y, pip_y = (0.5, 0.1), 0.75, 0.45, 0.5
    pts = [wrist] * 21
    for name, (tip, pip, x) in _FINGERS.items():
        pts[pip] = (x, pip_y)
        pts[tip] = (x, ext_y if name in extended else curl_y)
    return pts


def _hand(extended: set[str], pointing: str = "up") -> Hand:
    return Hand(id=1, bbox=(0, 0, 10, 10), landmarks=_landmarks(extended, pointing))


def test_one_finger_is_one():
    assert recognize_gesture(_hand({"index"})) == "one"


def test_two_fingers_is_two():
    assert recognize_gesture(_hand({"index", "middle"})) == "two"


def test_no_fingers_is_fist():
    assert recognize_gesture(_hand(set())) == "fist"


def test_three_fingers_is_other():
    assert recognize_gesture(_hand({"index", "middle", "ring"})) == "other"


def test_middle_only_is_other():
    assert recognize_gesture(_hand({"middle"})) == "other"


def test_missing_landmarks_is_other():
    assert recognize_gesture(Hand(id=1, bbox=(0, 0, 10, 10), landmarks=[])) == "other"


def test_orientation_down_still_recognized():
    assert recognize_gesture(_hand({"index"}, pointing="down")) == "one"
    assert recognize_gesture(_hand(set(), pointing="down")) == "fist"
