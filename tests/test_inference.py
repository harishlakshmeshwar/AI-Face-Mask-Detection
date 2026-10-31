"""Tests for CNN inference orchestration with a mocked model."""

import numpy as np
import pytest

from src.inference import MaskClassifier, ModelNotFoundError
from src.preprocessing import InvalidImageError


def test_classifier_reads_labels_and_size(classifier_mask, labels_file):
    assert classifier_mask.class_names == ["with_mask", "without_mask"]
    assert classifier_mask.image_size == 64
    assert classifier_mask.is_loaded


def test_predict_face_maps_probabilities_to_label(classifier_mask, face_crop):
    result = classifier_mask.predict_face(face_crop)
    assert result["label"] == "with_mask"
    assert result["confidence"] == pytest.approx(0.92)
    assert set(result["probabilities"]) == {"with_mask", "without_mask"}


def test_predict_face_no_mask_model(classifier_no_mask, face_crop):
    result = classifier_no_mask.predict_face(face_crop)
    assert result["label"] == "without_mask"
    assert result["confidence"] == pytest.approx(0.90)


def test_predict_face_requires_model(labels_file, tmp_path):
    classifier = MaskClassifier(
        model_path=tmp_path / "missing.keras",
        labels_path=labels_file,
        model=None,
    )
    # Force the unloaded state (load() already failed on missing files).
    classifier.model = None
    classifier.load_error = "Model is not loaded."
    with pytest.raises(ModelNotFoundError):
        classifier.predict_face(np.zeros((10, 10, 3), dtype=np.uint8))


def test_missing_model_files_report_load_error(tmp_path, labels_file):
    classifier = MaskClassifier(
        model_path=tmp_path / "missing.keras",
        labels_path=labels_file,
    )
    assert classifier.is_loaded is False
    assert classifier.load_error is not None
    assert "Model file not found" in classifier.load_error


def test_missing_labels_file_reports_load_error(tmp_path):
    from src.detector import FaceDetector

    classifier = MaskClassifier(
        model_path=tmp_path / "no_model.keras",
        labels_path=tmp_path / "no_labels.json",
        detector=FaceDetector(),
    )
    assert classifier.is_loaded is False
    assert classifier.load_error is not None


def test_predict_frame_counts_and_detections(classifier_mask, blank_frame, monkeypatch):
    # Force the detector to report two fake face boxes so inference is tested
    # without depending on a real face in the frame.
    monkeypatch.setattr(
        classifier_mask.detector,
        "detect",
        lambda frame: [(10, 10, 60, 60), (100, 40, 50, 50)],
    )
    detections, counts = classifier_mask.predict_frame(blank_frame)
    assert len(detections) == 2
    assert counts == {"faces": 2, "with_mask": 2, "without_mask": 0}
    assert detections[0]["bbox"] == [10, 10, 60, 60]
    assert detections[0]["display_label"] == "Mask"
    assert 0.0 <= detections[0]["confidence"] <= 1.0


def test_predict_frame_no_faces(classifier_mask, blank_frame, monkeypatch):
    monkeypatch.setattr(classifier_mask.detector, "detect", lambda frame: [])
    detections, counts = classifier_mask.predict_frame(blank_frame)
    assert detections == []
    assert counts["faces"] == 0


def test_predict_frame_rejects_invalid_input(classifier_mask):
    with pytest.raises(InvalidImageError):
        classifier_mask.predict_frame(None)


def test_annotate_draws_on_copy(classifier_mask, blank_frame):
    annotated = classifier_mask.annotate(
        blank_frame, [{"bbox": [5, 5, 40, 40], "label": "with_mask", "display_label": "Mask", "confidence": 0.9}]
    )
    assert annotated.shape == blank_frame.shape
    assert not np.array_equal(annotated, blank_frame)
    # Original frame must not be mutated.
    assert blank_frame[5, 5].tolist() == [30, 30, 30]


def test_process_frame_end_to_end_with_mock(classifier_mask, blank_frame, monkeypatch):
    monkeypatch.setattr(classifier_mask.detector, "detect", lambda frame: [(20, 20, 80, 80)])
    annotated, detections, counts, latency = classifier_mask.process_frame(blank_frame)
    assert annotated.shape == blank_frame.shape
    assert counts["faces"] == 1
    assert isinstance(latency, float) and latency >= 0.0
