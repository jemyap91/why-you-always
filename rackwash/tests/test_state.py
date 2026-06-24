from rackwash.state import SharedState


def test_initial_snapshot_is_empty_and_offline():
    state = SharedState()
    snap = state.snapshot()
    assert snap["frame_jpeg"] is None
    assert snap["camera_online"] is False
    assert snap["counts"] == {}


def test_publish_then_snapshot_reflects_latest():
    state = SharedState()
    counts = {"today": {"You": 1, "Wife": 0}}
    state.publish(b"jpegbytes", counts, camera_online=True)
    snap = state.snapshot()
    assert snap["frame_jpeg"] == b"jpegbytes"
    assert snap["counts"]["today"]["You"] == 1
    assert snap["camera_online"] is True


def test_snapshot_counts_are_copied_not_aliased():
    state = SharedState()
    state.publish(None, {"today": {"You": 1}}, camera_online=True)
    snap = state.snapshot()
    snap["counts"]["today"] = "tampered"
    assert state.snapshot()["counts"]["today"] == {"You": 1}


def test_latest_jpeg_returns_published_bytes():
    state = SharedState()
    state.publish(b"abc", {}, camera_online=True)
    assert state.latest_jpeg() == b"abc"
