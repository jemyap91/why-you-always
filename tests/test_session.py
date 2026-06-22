from dishcounter.session import SessionController


def test_gesture_held_long_enough_sets_washer():
    s = SessionController(hold_seconds=1.0)
    assert s.update("one", 0.0) is None       # just started holding
    assert s.update("one", 0.5) is None        # not held long enough
    assert s.update("one", 1.0) == "You"       # held 1.0s -> acts
    assert s.active == "You"


def test_two_fingers_sets_wife_and_fist_ends():
    s = SessionController(hold_seconds=1.0)
    s.update("two", 0.0)
    assert s.update("two", 1.0) == "Wife"
    s.update("fist", 1.0)
    assert s.update("fist", 2.0) is None       # fist held 1.0s -> session ends
    assert s.active is None


def test_brief_gesture_does_not_act():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    s.update("other", 0.3)                      # dropped before the hold elapsed
    assert s.update("one", 0.6) is None         # timer restarted at 0.6
    assert s.active is None


def test_changing_gesture_resets_the_timer():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    s.update("two", 0.5)                         # switched candidate
    assert s.update("two", 1.0) is None          # only 0.5s on 'two'
    assert s.update("two", 1.5) == "Wife"


def test_same_gesture_acts_once_then_is_idempotent():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("one", 1.0) == "You"
    # Still holding 'one' later must not re-trigger anything new.
    assert s.update("one", 5.0) == "You"
    assert s.active == "You"
