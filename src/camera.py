"""Webcam access with explicit start/stop and guaranteed resource release."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

BUSY_HINT = (
    "The camera opened but delivered no frames. It is most likely in use by "
    "another application - close other camera tabs/apps (browser tabs with "
    "camera access, Teams, Zoom, Camera) and try again."
)


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class WebcamCamera:
    """Thread-safe wrapper around ``cv2.VideoCapture``.

    Always release the camera with :meth:`stop` (or a ``with`` block) so the
    operating system regains exclusive access to the webcam.

    Windows webcams are exclusive: OpenCV can report "opened" while the
    device is held by another process, so :meth:`start` verifies that a
    real frame is delivered before declaring success.
    """

    def __init__(self, source: int | str = 0) -> None:
        self.source = source
        self._capture: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

    @property
    def is_opened(self) -> bool:
        with self._lock:
            return self._capture is not None and self._capture.isOpened()

    def _read_unlocked(self, attempts: int = 5, delay: float = 0.15) -> tuple[bool, Optional[np.ndarray]]:
        if self._capture is None or not self._capture.isOpened():
            return False, None
        for _ in range(max(attempts, 1)):
            ok, frame = self._capture.read()
            if ok and frame is not None and getattr(frame, "size", 0) > 0:
                return True, frame
            time.sleep(delay)
        return False, None

    def start(self) -> "WebcamCamera":
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return self
            try:
                capture = cv2.VideoCapture(self.source)
            except Exception as exc:
                raise CameraError(f"Could not access camera source '{self.source}': {exc}") from exc

            if not capture.isOpened():
                capture.release()
                raise CameraError(
                    f"Camera source '{self.source}' is unavailable or permission was denied. "
                    "Check that a webcam is connected, not disabled in Windows "
                    "Settings > Privacy > Camera, and not in use by another application."
                )

            # Modest buffers keep latency low for live preview.
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self._capture = capture
            ok, _frame = self._read_unlocked()
            if not ok:
                capture.release()
                self._capture = None
                raise CameraError(BUSY_HINT)
            return self

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return self._read_unlocked()

    def stop(self) -> None:
        with self._lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None

    def __enter__(self) -> "WebcamCamera":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
