"""Shared pytest fixtures for the AI Face Mask Detection test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import MaskClassifier  # noqa: E402


class FakeModel:
    """Stand-in for a Keras model: always predicts the given probabilities."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        batch_size = int(np.asarray(batch).shape[0])
        return np.tile(np.array([self.probabilities], dtype=np.float32), (batch_size, 1))


@pytest.fixture
def labels_file(tmp_path):
    path = tmp_path / "class_labels.json"
    path.write_text(
        json.dumps({"labels": ["with_mask", "without_mask"], "image_size": 64}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_model_mask():
    return FakeModel([0.92, 0.08])


@pytest.fixture
def fake_model_no_mask():
    return FakeModel([0.10, 0.90])


@pytest.fixture
def classifier_mask(labels_file, fake_model_mask, tmp_path):
    return MaskClassifier(
        model_path=tmp_path / "unused.keras",
        labels_path=labels_file,
        confidence_threshold=0.5,
        model=fake_model_mask,
    )


@pytest.fixture
def classifier_no_mask(labels_file, fake_model_no_mask, tmp_path):
    return MaskClassifier(
        model_path=tmp_path / "unused.keras",
        labels_path=labels_file,
        confidence_threshold=0.5,
        model=fake_model_no_mask,
    )


@pytest.fixture
def blank_frame():
    """A 240x320 BGR frame."""
    return np.full((240, 320, 3), 30, dtype=np.uint8)


@pytest.fixture
def face_crop():
    """A synthetic 80x80 BGR 'face' crop (skin-ish tone)."""
    crop = np.zeros((80, 80, 3), dtype=np.uint8)
    crop[:, :] = (140, 170, 210)  # BGR skin tone approximation
    return crop
