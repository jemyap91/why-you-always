from fastapi.testclient import TestClient

from rackwash.dashboard import create_app
from rackwash.state import SharedState


def test_index_serves_html():
    client = TestClient(create_app(SharedState()))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Dish Counter" in resp.text


def test_index_has_zoom_controls():
    client = TestClient(create_app(SharedState()))
    resp = client.get("/")
    assert "zoom(" in resp.text  # client-side zoom buttons wired up
    assert 'id="feed"' in resp.text


def test_counts_endpoint_reflects_state():
    state = SharedState()
    state.publish(b"x", {"today": {"You": 3, "Wife": 1}}, camera_online=True)
    client = TestClient(create_app(state))
    resp = client.get("/counts")
    assert resp.status_code == 200
    assert resp.json()["today"] == {"You": 3, "Wife": 1}


def test_websocket_pushes_counts_then_can_close():
    state = SharedState()
    state.publish(None, {"all_time": {"You": 2, "Wife": 2}}, camera_online=False)
    client = TestClient(create_app(state))
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["counts"]["all_time"] == {"You": 2, "Wife": 2}
        assert msg["camera_online"] is False


def test_reset_endpoint_calls_store_reset():
    class FakeStore:
        def __init__(self):
            self.reset_calls = 0

        def reset(self, now):
            self.reset_calls += 1

    store = FakeStore()
    client = TestClient(create_app(SharedState(), store))
    assert client.post("/reset").status_code == 200
    assert store.reset_calls == 1


def test_reset_endpoint_ok_without_store():
    client = TestClient(create_app(SharedState()))
    assert client.post("/reset").status_code == 200


def test_detections_endpoint_toggles_flag():
    state = SharedState()
    client = TestClient(create_app(state))
    assert state.show_detections() is False
    client.post("/detections")
    assert state.show_detections() is True


def test_index_has_reset_and_detections_buttons():
    text = TestClient(create_app(SharedState())).get("/").text
    assert "/reset" in text
    assert "/detections" in text


def test_ws_includes_detection_fields():
    state = SharedState()
    state.publish(None, {"all_time": {"You": 0, "Wife": 0}}, camera_online=True,
                  detections=["plate"])
    client = TestClient(create_app(state))
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert "show_detections" in msg
        assert msg["detections"] == ["plate"]
