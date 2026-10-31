"""Lightweight face detection using OpenCV Haar cascades."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .preprocessing import validate_image

BoundingBox = Tuple[int, int, int, int]


class DetectorError(RuntimeError):
    """Raised when the face detector cannot be initialised."""


class FaceDetector:
    """Detects frontal faces with the Haar cascade bundled inside opencv-python."""

    def __init__(
        self,
        cascade_path: str | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40),
    ) -> None:
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        path = cascade_path or cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade_path = path
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise DetectorError(
                f"Failed to load Haar cascade from '{path}'. "
                "Reinstall opencv-python or provide a valid cascade path."
            )

    @property
    def available(self) -> bool:
        return self._cascade is not None and not self._cascade.empty()

    def detect(self, frame: np.ndarray | None) -> List[BoundingBox]:
        """Return a list of (x, y, w, h) face boxes for a BGR frame."""
        if frame is None:
            return []
        frame = validate_image(frame)

        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces: Sequence[Tuple[int, int, int, int]] = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
