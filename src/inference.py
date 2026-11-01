"""CNN inference: classify detected faces as Mask / No Mask."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .detector import FaceDetector
from .preprocessing import InvalidImageError, to_model_input, validate_image

MASK_COLOR_BGR = (76, 175, 80)
NO_MASK_COLOR_BGR = (66, 66, 235)
BOX_THICKNESS = 2

# BGR colours keyed by class name; unknown classes fall back to grey.
CLASS_COLORS_BGR = {
    "with_mask": MASK_COLOR_BGR,
    "without_mask": NO_MASK_COLOR_BGR,
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model or its label configuration is missing."""


class MaskClassifier:
    """Loads a trained Keras model and classifies face crops or full frames.

    A ``model`` (any object with ``predict``) can be injected for testing so
    unit tests never need real weights or a webcam.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        confidence_threshold: float | None = None,
        detector: FaceDetector | None = None,
        model: Any = None,
    ) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.labels_path = Path(labels_path or config.CLASS_LABELS_PATH)
        self.confidence_threshold = (
            config.CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        )
        self.detector = detector or FaceDetector()
        self.model = model
        self.class_names: List[str] = list(config.CLASS_NAMES)
        self.image_size: int = config.DEFAULT_IMAGE_SIZE
        self.load_error: Optional[str] = None

        if model is not None:
            self._read_labels()
        else:
            self.load()

    # ------------------------------------------------------------------ setup
    def _read_labels(self) -> None:
        if self.labels_path.exists():
            with open(self.labels_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            labels = payload.get("labels") or config.CLASS_NAMES
            self.class_names = [str(name) for name in labels]
            self.image_size = int(payload.get("image_size", self.image_size))

    def load(self) -> bool:
        """Load model weights and label configuration. Returns success flag."""
        self.load_error = None
        try:
            if not self.model_path.exists():
                raise ModelNotFoundError(
                    f"Model file not found at '{self.model_path}'. "
                    "Train a model first: python -m training.train --data-dir dataset"
                )
            if not self.labels_path.exists():
                raise ModelNotFoundError(
                    f"Label configuration not found at '{self.labels_path}'. "
                    "It is written automatically by the training script."
                )
            import tensorflow as tf  # imported lazily so tests can mock the model

            self._read_labels()
            self.model = tf.keras.models.load_model(self.model_path)
            return True
        except ModelNotFoundError as exc:
            self.load_error = str(exc)
            self.model = None
            return False
        except Exception as exc:  # corrupt/incompatible model file
            self.load_error = f"Failed to load model: {exc}"
            self.model = None
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- inference
    def predict_face(self, bgr_face: np.ndarray) -> Dict[str, Any]:
        """Classify a single BGR face crop."""
        if self.model is None:
            raise ModelNotFoundError(self.load_error or "Model is not loaded.")

        tensor = to_model_input(bgr_face, self.image_size)
        batch = tensor[np.newaxis, ...]
        probs = np.asarray(self.model.predict(batch, verbose=0))[0]
        best = int(np.argmax(probs))
        return {
            "label": self.class_names[best],
            "confidence": float(probs[best]),
            "probabilities": {
                name: float(probs[index]) for index, name in enumerate(self.class_names)
            },
        }

    def predict_frame(self, bgr_frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Detect faces in a frame and classify each one.

        Returns ``(detections, counts)``. Each detection carries the bounding
        box, class label and confidence. Frames are never classified whole:
        every face crop is preprocessed and run through the CNN individually.
        """
        frame = validate_image(bgr_frame)
        boxes = self.detector.detect(frame)

        detections: List[Dict[str, Any]] = []
        for index, (x, y, w, h) in enumerate(boxes):
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue
            try:
                prediction = self.predict_face(crop)
            except InvalidImageError:
                continue
            detections.append(
                {
                    "id": index + 1,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "label": prediction["label"],
                    "display_label": config.DISPLAY_NAMES.get(
                        prediction["label"], prediction["label"]
                    ),
                    "confidence": round(float(prediction["confidence"]), 4),
                    "probabilities": {
                        key: round(value, 4)
                        for key, value in prediction["probabilities"].items()
                    },
                }
            )

        counts = self.count_classes(detections)
        return detections, counts

    @staticmethod
    def count_classes(detections: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"faces": len(detections), "with_mask": 0, "without_mask": 0}
        for detection in detections:
            label = detection.get("label")
            if label in counts:
                counts[label] += 1
        return counts

    # ------------------------------------------------------------ visualisation
    def annotate(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes, labels and confidence onto a copy of the frame."""
        annotated = validate_image(frame).copy()
        for detection in detections:
            x, y, w, h = detection["bbox"]
            label = detection.get("label", "")
            confidence = float(detection.get("confidence", 0.0))
            color = CLASS_COLORS_BGR.get(label, (200, 200, 200))

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, BOX_THICKNESS)

            display = detection.get("display_label") or config.DISPLAY_NAMES.get(label, label)
            text = f"{display} {confidence * 100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            top = max(y - text_h - baseline - 6, 0)
            cv2.rectangle(annotated, (x, top), (x + text_w + 8, top + text_h + baseline + 6), color, -1)
            cv2.putText(
                annotated,
                text,
                (x + 4, top + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (18, 18, 18),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_frame(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int], float]:
        """Full pipeline for one frame: detect -> classify -> annotate.

        Returns ``(annotated_frame, detections, counts, latency_ms)``.
        """
        started = time.time()
        detections, counts = self.predict_frame(bgr_frame)
        annotated = self.annotate(bgr_frame, detections)
        latency_ms = round((time.time() - started) * 1000, 2)
        return annotated, detections, counts, latency_ms
