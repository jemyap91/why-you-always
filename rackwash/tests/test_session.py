from rackwash.session import SessionController


def test_one_starts_you_then_repeat_ends():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("one", 1.0) == "You"      # held 1s -> start You
    s.update("other", 1.5)                      # drop the gesture (break the hold)
    s.update("one", 2.0)
    assert s.update("one", 3.0) is None         # repeat -> end the session
    assert s.active is None


def test_two_starts_wife_then_repeat_ends():
    s = SessionController(hold_seconds=1.0)
    s.update("two", 0.0)
    assert s.update("two", 1.0) == "Wife"
    s.update("other", 1.5)
    s.update("two", 2.0)
    assert s.update("two", 3.0) is None


def test_showing_other_number_switches_directly():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("one", 1.0) == "You"
    s.update("two", 1.0)                         # switch candidate to 'two'
    assert s.update("two", 2.0) == "Wife"        # held -> switch to Wife
    assert s.active == "Wife"


def test_held_number_acts_once_and_stays_active_while_held():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("one", 1.0) == "You"
    # Keep holding 'one' -> must stay You, not flip back off every frame.
    assert s.update("one", 2.0) == "You"
    assert s.update("one", 5.0) == "You"
    assert s.active == "You"


def test_fist_does_nothing():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    s.update("one", 1.0)                         # You active
    s.update("fist", 1.0)
    assert s.update("fist", 2.0) == "You"        # fist held -> no effect
    assert s.active == "You"


def test_brief_gesture_does_not_act():
    s = SessionController(hold_seconds=1.0)
    s.update("one", 0.0)
    assert s.update("other", 0.5) is None        # dropped before the hold elapsed
    assert s.active is None
