"""Image loading and preprocessing shared by training and inference.

Colour contract: the CNN is trained on RGB images scaled to [0, 1] as float32
with shape (image_size, image_size, 3). OpenCV produces BGR frames, so every
path that feeds the model must call :func:`bgr_to_rgb` first.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


class InvalidImageError(ValueError):
    """Raised when an input cannot be interpreted as a usable image."""


def validate_image(image: np.ndarray | None) -> np.ndarray:
    """Return the image if it is a non-empty ndarray, else raise."""
    if image is None:
        raise InvalidImageError("Image is None.")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Expected numpy.ndarray, got {type(image).__name__}.")
    if image.size == 0:
        raise InvalidImageError("Image array is empty.")
    if image.ndim not in (2, 3):
        raise InvalidImageError(f"Unsupported image dimensions: {image.ndim}.")
    return image


def load_bgr_image(source: str | np.ndarray) -> np.ndarray:
    """Load an image from a file path or pass through an ndarray as BGR."""
    if isinstance(source, (str,)):
        image = cv2.imread(source, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError(f"Could not read image file: {source}")
        return image
    image = validate_image(source)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or grayscale) image to RGB."""
    image = validate_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_model_input(bgr_face: np.ndarray, image_size: int) -> np.ndarray:
    """Convert a BGR face crop into the normalized RGB tensor used by the CNN.

    Returns a float32 array of shape (image_size, image_size, 3) with values
    in [0, 1]. Identical normalisation must be used at training time.
    """
    if image_size <= 0:
        raise InvalidImageError("image_size must be a positive integer.")
    rgb = bgr_to_rgb(bgr_face)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    return np.clip(tensor, 0.0, 1.0)


def to_model_batch(bgr_faces: Iterable[np.ndarray], image_size: int) -> np.ndarray:
    """Stack :func:`to_model_input` results into a batch array."""
    faces = list(bgr_faces)
    if not faces:
        raise InvalidImageError("No face crops supplied for batching.")
    return np.stack([to_model_input(face, image_size) for face in faces], axis=0)


def tensor_to_model_input(rgb_tensor: np.ndarray, image_size: int) -> np.ndarray:
    """Convert an already-RGB image tensor (0-255 uint8 or float) for training.

    Used by the training pipeline so normalisation matches inference exactly.
    """
    tensor = validate_image(rgb_tensor)
    if tensor.ndim == 2:
        tensor = np.stack([tensor] * 3, axis=-1)
    if tensor.shape[-1] == 1:
        tensor = np.repeat(tensor, 3, axis=-1)
    if tensor.shape[-1] != 3:
        raise InvalidImageError(f"Expected 1, 3 or 4 channels, got {tensor.shape[-1]}.")
    tensor = tensor[..., :3].astype(np.float32)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    resized = cv2.resize(tensor, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.clip(resized.astype(np.float32), 0.0, 1.0)
