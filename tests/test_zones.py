import numpy as np

from dishcounter.config import SkinProfile, Zone
from dishcounter.domain import Hand
from dishcounter.identity import IdentityClassifier
from dishcounter.zones import ZoneEventEngine

SINK = Zone(x1=0, y1=0, x2=100, y2=100)
DRYING = Zone(x1=100, y1=0, x2=200, y2=100)

# Profiles chosen so a reddish region -> "You".
YOU = SkinProfile(cr=165.0, cb=110.0)
WIFE = SkinProfile(cr=120.0, cb=150.0)
RED = np.tile(np.array([80, 80, 220], dtype=np.uint8), (9, 1))  # reddish region


def _engine():
    clf = IdentityClassifier(YOU, WIFE, max_distance=60.0)
    return ZoneEventEngine(SINK, DRYING, clf, presence_window=5.0, cooldown=3.0)


def _hand_at(cx, region=RED, hand_id=None):
    return Hand(id=hand_id, bbox=(cx - 5, 45, cx + 5, 55), region_pixels=region)


def test_sink_to_drying_emits_one_event_for_you():
    eng = _engine()
    assert eng.process([_hand_at(50, hand_id=1)], now=0.0) == []  # in sink
    events = eng.process([_hand_at(150, hand_id=1)], now=1.0)     # now in drying
    assert len(events) == 1
    assert events[0].person == "You"
    assert events[0].hand_id == 1


def test_no_event_without_prior_sink_visit():
    eng = _engine()
    events = eng.process([_hand_at(150, hand_id=1)], now=1.0)  # appears in drying only
    assert events == []


def test_cooldown_blocks_immediate_refire():
    eng = _engine()
    eng.process([_hand_at(50, hand_id=1)], now=0.0)
    first = eng.process([_hand_at(150, hand_id=1)], now=1.0)
    assert len(first) == 1
    # Linger on the drying side / re-enter quickly: still within cooldown.
    eng.process([_hand_at(50, hand_id=1)], now=1.5)
    second = eng.process([_hand_at(150, hand_id=1)], now=2.0)
    assert second == []  # cooldown (3s) not elapsed


def test_refire_allowed_after_cooldown():
    eng = _engine()
    eng.process([_hand_at(50, hand_id=1)], now=0.0)
    eng.process([_hand_at(150, hand_id=1)], now=1.0)
    eng.process([_hand_at(50, hand_id=1)], now=5.0)
    again = eng.process([_hand_at(150, hand_id=1)], now=6.0)
    assert len(again) == 1


def test_stale_sink_visit_does_not_count():
    eng = _engine()
    eng.process([_hand_at(50, hand_id=1)], now=0.0)
    # Crossing happens long after the presence window (5s) expired.
    events = eng.process([_hand_at(150, hand_id=1)], now=10.0)
    assert events == []


def test_uncertain_identity_still_emits_but_tagged_uncertain():
    eng = _engine()
    green = np.tile(np.array([20, 230, 20], dtype=np.uint8), (9, 1))  # far from both
    eng.process([_hand_at(50, region=green, hand_id=1)], now=0.0)
    events = eng.process([_hand_at(150, region=green, hand_id=1)], now=1.0)
    assert len(events) == 1
    assert events[0].person == "uncertain"


def test_two_hands_cross_independently():
    eng = _engine()
    blue = np.tile(np.array([220, 90, 90], dtype=np.uint8), (9, 1))  # -> Wife
    eng.process(
        [_hand_at(50, hand_id=1), _hand_at(40, region=blue, hand_id=2)], now=0.0
    )
    events = eng.process(
        [_hand_at(150, hand_id=1), _hand_at(160, region=blue, hand_id=2)], now=1.0
    )
    assert {e.person for e in events} == {"You", "Wife"}


def test_lingering_in_drying_after_fire_does_not_double_count():
    eng = _engine()
    eng.process([_hand_at(50, hand_id=1)], now=0.0)        # in sink
    first = eng.process([_hand_at(150, hand_id=1)], now=1.0)  # crosses -> 1 event
    assert len(first) == 1
    # Hand stays in the drying zone on the next frame: must NOT re-fire.
    second = eng.process([_hand_at(150, hand_id=1)], now=1.5)
    assert second == []


def test_refire_at_exact_cooldown_boundary_is_allowed():
    eng = _engine()
    eng.process([_hand_at(50, hand_id=1)], now=0.0)
    eng.process([_hand_at(150, hand_id=1)], now=1.0)        # fires at t=1.0
    # Re-enter sink, then cross exactly cooldown (3.0s) after the fire: allowed.
    eng.process([_hand_at(50, hand_id=1)], now=3.5)
    again = eng.process([_hand_at(150, hand_id=1)], now=4.0)  # 4.0 - 1.0 == 3.0 == cooldown
    assert len(again) == 1
