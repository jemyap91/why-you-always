import numpy as np

from dishcounter.config import SkinProfile
from dishcounter.domain import Hand, median_chroma
from dishcounter.identity import IdentityClassifier


def _hand_from_bgr(color_bgr):
    pixels = np.tile(np.array(color_bgr, dtype=np.uint8), (9, 1))
    return Hand(id=1, bbox=(0, 0, 3, 3), region_pixels=pixels)


def test_no_pixels_is_uncertain():
    clf = IdentityClassifier(SkinProfile(cr=150, cb=110), SkinProfile(cr=140, cb=120))
    label, conf = clf.classify(Hand(id=1, bbox=(0, 0, 1, 1), region_pixels=None))
    assert label == "uncertain"
    assert conf == 0.0


def test_classifies_to_nearest_profile():
    you_color = (100, 100, 200)   # reddish -> high Cr
    you_cr, you_cb = median_chroma(np.array([you_color], dtype=np.uint8))
    wife_color = (200, 100, 100)  # bluish -> high Cb
    wife_cr, wife_cb = median_chroma(np.array([wife_color], dtype=np.uint8))
    clf = IdentityClassifier(
        SkinProfile(cr=you_cr, cb=you_cb),
        SkinProfile(cr=wife_cr, cb=wife_cb),
    )
    label, conf = clf.classify(_hand_from_bgr(you_color))
    assert label == "You"
    assert conf > 0.0


def test_far_from_both_profiles_is_uncertain():
    clf = IdentityClassifier(
        SkinProfile(cr=150, cb=110), SkinProfile(cr=160, cb=110), max_distance=10.0
    )
    label, _ = clf.classify(_hand_from_bgr((10, 240, 10)))  # green: chroma far from both
    assert label == "uncertain"


def test_ambiguous_between_profiles_is_uncertain():
    # Two profiles equidistant from the sample -> ambiguous -> uncertain.
    clf = IdentityClassifier(
        SkinProfile(cr=128, cb=120), SkinProfile(cr=128, cb=136), max_distance=50.0
    )
    gray = _hand_from_bgr((100, 100, 100))  # chroma ~ (128,128), equidistant in Cb
    label, _ = clf.classify(gray)
    assert label == "uncertain"
