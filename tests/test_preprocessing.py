"""Tests for shared image preprocessing and colour handling."""

import numpy as np
import pytest

from src.preprocessing import (
    InvalidImageError,
    bgr_to_rgb,
    load_bgr_image,
    tensor_to_model_input,
    to_model_batch,
    to_model_input,
    validate_image,
)


def test_validate_image_rejects_none():
    with pytest.raises(InvalidImageError):
        validate_image(None)


def test_validate_image_rejects_empty_array():
    with pytest.raises(InvalidImageError):
        validate_image(np.array([]))


def test_validate_image_accepts_frame(blank_frame):
    assert validate_image(blank_frame) is blank_frame


def test_bgr_to_rgb_swaps_channels():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # blue channel in BGR
    rgb = bgr_to_rgb(bgr)
    assert rgb[:, :, 2].max() == 255
    assert rgb[:, :, 0].max() == 0


def test_to_model_input_shape_and_range(face_crop):
    tensor = to_model_input(face_crop, 64)
    assert tensor.shape == (64, 64, 3)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() <= tensor.max() <= 1.0


def test_to_model_input_preserves_colour_semantics():
    """A pure-blue BGR pixel must arrive in the model's RGB blue channel."""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255
    tensor = to_model_input(bgr, 16)
    assert tensor[..., 2].mean() == pytest.approx(1.0)
    assert tensor[..., 0].mean() == pytest.approx(0.0)


def test_to_model_input_rejects_bad_size(face_crop):
    with pytest.raises(InvalidImageError):
        to_model_input(face_crop, 0)


def test_to_model_batch_stacks():
    batch = to_model_batch([np.zeros((20, 20, 3), np.uint8)] * 3, 32)
    assert batch.shape == (3, 32, 32, 3)


def test_to_model_batch_empty_raises():
    with pytest.raises(InvalidImageError):
        to_model_batch([], 32)


def test_load_bgr_image_missing_file(tmp_path):
    with pytest.raises(InvalidImageError):
        load_bgr_image(str(tmp_path / "nope.jpg"))


def test_load_bgr_image_grayscale_becomes_bgr():
    gray = np.full((8, 8), 120, dtype=np.uint8)
    bgr = load_bgr_image(gray)
    assert bgr.ndim == 3 and bgr.shape[2] == 3


def test_tensor_to_model_input_normalizes_float_images():
    rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
    tensor = tensor_to_model_input(rgb, 24)
    assert tensor.shape == (24, 24, 3)
    assert tensor.max() == pytest.approx(1.0)


def test_tensor_to_model_input_keeps_already_scaled_floats():
    rgb = np.ones((20, 20, 3), dtype=np.float32)
    tensor = tensor_to_model_input(rgb, 24)
    assert tensor.max() == pytest.approx(1.0)
