"""Tests for webcam open/stop behaviour (no physical camera required)."""

import numpy as np
import pytest

from src.camera import BUSY_HINT, CameraError, WebcamCamera


class FakeCapture:
    """Stands in for cv2.VideoCapture with configurable behaviour."""

    def __init__(self, opened=True, delivers_frames=True):
        self._opened = opened
        self._delivers = delivers_frames
        self.released = False

    def isOpened(self):
        return self._opened and not self.released

    def read(self):
        if not self._delivers:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def set(self, *args, **kwargs):
        return True

    def release(self):
        self.released = True


def test_start_raises_when_open_fails(monkeypatch):
    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda source: FakeCapture(opened=False))
    camera = WebcamCamera(0)
    with pytest.raises(CameraError):
        camera.start()
    assert camera.is_opened is False


def test_start_raises_when_no_frames_delivered(monkeypatch):
    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda source: FakeCapture(delivers_frames=False))
    camera = WebcamCamera(0)
    with pytest.raises(CameraError) as excinfo:
        camera.start()
    assert BUSY_HINT in str(excinfo.value)
    # The capture must be released so another app can use the webcam.
    assert camera.is_opened is False


def test_start_success_and_stop_releases(monkeypatch):
    fake = FakeCapture()
    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda source: fake)
    camera = WebcamCamera(0)
    camera.start()
    assert camera.is_opened is True
    ok, frame = camera.read()
    assert ok and frame is not None
    camera.stop()
    assert camera.is_opened is False
    assert fake.released is True


def test_read_without_start(monkeypatch):
    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda source: FakeCapture())
    camera = WebcamCamera(0)
    ok, frame = camera.read()
    assert ok is False and frame is None


def test_context_manager_releases(monkeypatch):
    fake = FakeCapture()
    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda source: fake)
    with WebcamCamera(0) as camera:
        assert camera.is_opened is True
    assert fake.released is True
