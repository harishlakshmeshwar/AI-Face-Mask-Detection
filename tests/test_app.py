"""API tests for the Flask application (mocked model, no webcam)."""

import base64

import numpy as np
import pytest

from app import create_app, decode_base64_frame
from src.inference import InvalidImageError


@pytest.fixture
def client(classifier_mask):
    app = create_app(classifier=classifier_mask)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def jpeg_data_url(value=120):
    """Build a tiny JPEG data-URL for upload tests."""
    import cv2

    frame = np.full((64, 64, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Face Mask Detection System" in response.data


def test_health_reports_loaded_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "online"
    assert payload["model"] == "loaded"
    assert payload["classes"] == ["with_mask", "without_mask"]


def test_detect_missing_model_returns_503(tmp_path, labels_file):
    from src.inference import MaskClassifier

    classifier = MaskClassifier(
        model_path=tmp_path / "missing.keras",
        labels_path=labels_file,
    )
    app = create_app(classifier=classifier)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        response = test_client.post("/api/detect", json={"image": jpeg_data_url()})
    assert response.status_code == 503
    assert response.get_json()["status"] == "model_missing"


def test_detect_rejects_invalid_payload(client):
    response = client.post("/api/detect", json={})
    assert response.status_code == 400


def test_detect_rejects_garbage_base64(client):
    response = client.post("/api/detect", json={"image": "data:image/jpeg;base64,%%%not-base64%%%"})
    assert response.status_code == 400


def test_detect_returns_faces_and_counts(client, monkeypatch):
    response = client.post("/api/detect", json={"image": jpeg_data_url()})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert set(payload["counts"]) == {"faces", "with_mask", "without_mask"}
    assert isinstance(payload["faces"], list)
    assert isinstance(payload["latency_ms"], float)


def test_detect_with_forced_face(client, monkeypatch):
    app = client.application
    classifier = app.extensions["session"].classifier
    monkeypatch.setattr(classifier.detector, "detect", lambda frame: [(5, 5, 30, 30)])
    response = client.post("/api/detect", json={"image": jpeg_data_url()})
    payload = response.get_json()
    assert payload["counts"]["faces"] == 1
    assert payload["faces"][0]["label"] == "with_mask"


def test_events_endpoints(client):
    reset = client.post("/api/events/reset")
    assert reset.status_code == 200
    events = client.get("/api/events").get_json()
    assert events["summary"]["total_detections"] == 0


def test_camera_stop_when_never_started(client):
    response = client.post("/api/camera/stop")
    assert response.status_code == 200
    assert response.get_json()["status"] == "stopped"


def test_detections_endpoint_defaults(client):
    response = client.get("/api/detections")
    payload = response.get_json()
    assert payload["camera_active"] is False
    assert payload["counts"]["faces"] == 0


def test_unknown_api_route_returns_json_404(client):
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["status"] == "error"


def test_decode_base64_frame_plain_base64():
    import cv2

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    decoded = decode_base64_frame(b64)
    assert decoded.shape == (10, 10, 3)


def test_decode_base64_frame_rejects_empty():
    with pytest.raises(InvalidImageError):
        decode_base64_frame("")
