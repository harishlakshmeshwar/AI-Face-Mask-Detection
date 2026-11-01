"""Tests for the face detector."""

import numpy as np
import pytest

from src.detector import DetectorError, FaceDetector


def test_detector_loads_bundled_cascade():
    detector = FaceDetector()
    assert detector.available


def test_detector_missing_cascade_raises(tmp_path):
    with pytest.raises(DetectorError):
        FaceDetector(cascade_path=str(tmp_path / "missing_cascade.xml"))


def test_detect_none_returns_empty():
    detector = FaceDetector()
    assert detector.detect(None) == []


def test_detect_on_blank_frame_returns_list(blank_frame):
    detector = FaceDetector()
    faces = detector.detect(blank_frame)
    assert isinstance(faces, list)


def test_detect_boxes_are_int_tuples(blank_frame):
    detector = FaceDetector()
    for box in detector.detect(blank_frame):
        assert len(box) == 4
        assert all(isinstance(value, int) for value in box)
