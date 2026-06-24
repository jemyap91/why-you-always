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
