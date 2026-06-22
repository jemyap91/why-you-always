import numpy as np

from dishcounter.camera import FakeCamera


def test_fake_camera_returns_frames_then_none():
    f1 = np.zeros((4, 4, 3), dtype=np.uint8)
    f2 = np.ones((4, 4, 3), dtype=np.uint8)
    cam = FakeCamera([f1, f2])
    assert np.array_equal(cam.read(), f1)
    assert np.array_equal(cam.read(), f2)
    assert cam.read() is None


def test_fake_camera_release_is_safe():
    cam = FakeCamera([])
    cam.release()  # must not raise
    assert cam.read() is None
