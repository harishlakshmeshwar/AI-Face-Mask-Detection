"""Flask application for the AI Face Mask Detection System.

Run locally::

    python app.py

Two detection modes are supported:

* **Browser camera** - the frontend captures frames with getUserMedia and
  posts them to ``POST /api/detect``. Works anywhere the browser allows
  camera access, including deployed frontends paired with this API.
* **Server camera** - this process opens the webcam with OpenCV
  (``POST /api/camera/start``), runs detection on a background thread and
  serves an annotated MJPEG stream at ``GET /video_feed``.
"""

from __future__ import annotations

import atexit
import base64
import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from src import config
from src.camera import CameraError, WebcamCamera
from src.detector import DetectorError, FaceDetector
from src.inference import MaskClassifier, ModelNotFoundError
from src.preprocessing import InvalidImageError

MAX_EVENT_HISTORY = config.EVENT_HISTORY_LIMIT
JPEG_QUALITY = 80


def decode_base64_frame(payload: str) -> np.ndarray:
    """Decode a data-URL or raw base64 JPEG/PNG into a BGR frame."""
    if not payload or not isinstance(payload, str):
        raise InvalidImageError("Missing image payload.")
    if "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        buffer = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    except Exception as exc:
        raise InvalidImageError("Image payload is not valid base64.") from exc
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise InvalidImageError("Could not decode image payload.")
    return frame


class DetectionSession:
    """Holds the classifier plus the optional server-camera worker."""

    def __init__(self, classifier: MaskClassifier) -> None:
        self.classifier = classifier
        self.camera = WebcamCamera(config.CAMERA_SOURCE)
        self.camera_active = False
        self.camera_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._last_detections: Dict[str, Any] = {
            "faces": [],
            "counts": {"faces": 0, "with_mask": 0, "without_mask": 0},
            "latency_ms": 0.0,
            "timestamp": None,
        }
        self.events = deque(maxlen=MAX_EVENT_HISTORY)

    # ----------------------------------------------------------- model state
    @property
    def model_loaded(self) -> bool:
        return self.classifier.is_loaded

    # ------------------------------------------------------------- events
    def record_events(self, detections, counts, latency_ms: float) -> None:
        if not detections:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for detection in detections:
            self.events.appendleft(
                {
                    "time": timestamp,
                    "label": detection["label"],
                    "display_label": detection["display_label"],
                    "confidence": detection["confidence"],
                    "bbox": detection["bbox"],
                    "source": "server_camera",
                    "latency_ms": latency_ms,
                }
            )

    def session_summary(self) -> Dict[str, Any]:
        events = list(self.events)
        total = len(events)
        with_mask = sum(1 for event in events if event["label"] == "with_mask")
        without_mask = total - with_mask
        return {
            "total_detections": total,
            "with_mask": with_mask,
            "without_mask": without_mask,
            "mask_ratio": round(with_mask / total, 4) if total else None,
            "events": events[:50],
        }

    def reset_events(self) -> None:
        self.events.clear()

    # ------------------------------------------------------- server camera
    def start_camera(self) -> Dict[str, Any]:
        if self.camera_active and self.camera.is_opened:
            return {"status": "running", "source": str(config.CAMERA_SOURCE)}
        try:
            self.camera.start()
        except CameraError as exc:
            self.camera_error = str(exc)
            raise

        self.camera_error = None
        self.camera_active = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="camera-worker", daemon=True)
        self._thread.start()
        return {"status": "running", "source": str(config.CAMERA_SOURCE)}

    def stop_camera(self) -> Dict[str, Any]:
        self.camera_active = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        self.camera.stop()
        self._latest_jpeg = None
        self._last_detections = {
            "faces": [],
            "counts": {"faces": 0, "with_mask": 0, "without_mask": 0},
            "latency_ms": 0.0,
            "timestamp": None,
        }
        return {"status": "stopped"}

    def _worker(self) -> None:
        interval = 1.0 / max(config.CAMERA_FPS, 1.0)
        while not self._stop_event.is_set():
            started = time.time()
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self.camera_error = (
                    "Camera stopped delivering frames. It may have been "
                    "disconnected or claimed by another application."
                )
                time.sleep(0.5)
                continue

            if not self.classifier.is_loaded:
                self._latest_jpeg = encode_jpeg(frame)
                time.sleep(interval)
                continue

            try:
                annotated, detections, counts, latency = self.classifier.process_frame(frame)
            except Exception as exc:
                self.camera_error = f"Detection error: {exc}"
                time.sleep(interval)
                continue

            with self._lock:
                self._latest_jpeg = encode_jpeg(annotated)
                self._last_detections = {
                    "faces": detections,
                    "counts": counts,
                    "latency_ms": latency,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            self.record_events(detections, counts, latency)

            elapsed = time.time() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_detections(self) -> Dict[str, Any]:
        with self._lock:
            payload = dict(self._last_detections)
        payload["camera_active"] = self.camera_active
        payload["camera_error"] = self.camera_error
        payload["model_loaded"] = self.model_loaded
        return payload


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return encoded.tobytes()


def create_app(
    classifier: Optional[MaskClassifier] = None,
    detector: Optional[FaceDetector] = None,
) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    if classifier is None:
        try:
            classifier = MaskClassifier(detector=detector)
        except DetectorError as exc:
            app.logger.error("Face detector unavailable: %s", exc)
            raise

    session = DetectionSession(classifier)
    app.extensions["session"] = session

    def shutdown_camera() -> None:
        try:
            session.stop_camera()
        except Exception:
            pass

    atexit.register(shutdown_camera)

    # ------------------------------------------------------------- pages
    @app.route("/")
    def index():
        return render_template("index.html")

    # -------------------------------------------------------------- api
    @app.route("/api/health")
    def health():
        payload = {
            "status": "online",
            "model": "loaded" if session.model_loaded else "missing",
            "model_error": classifier.load_error,
            "classes": classifier.class_names,
            "image_size": classifier.image_size,
            "confidence_threshold": classifier.confidence_threshold,
            "detector": {
                "available": classifier.detector.available,
                "cascade": classifier.detector.cascade_path,
            },
            "camera_source": str(config.CAMERA_SOURCE),
            "camera_active": session.camera_active,
        }
        return jsonify(payload)

    @app.route("/api/detect", methods=["POST"])
    def detect():
        if not session.model_loaded:
            return (
                jsonify(
                    {
                        "status": "model_missing",
                        "message": classifier.load_error
                        or "No trained model found. Train one with: python -m training.train --data-dir dataset",
                    }
                ),
                503,
            )

        data = request.get_json(silent=True) or {}
        image_payload = data.get("image")
        try:
            frame = decode_base64_frame(image_payload)
        except InvalidImageError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        try:
            _, detections, counts, latency = classifier.process_frame(frame)
        except InvalidImageError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

        session.record_events(detections, counts, latency)
        return jsonify(
            {
                "status": "success",
                "faces": detections,
                "counts": counts,
                "latency_ms": latency,
            }
        )

    @app.route("/api/detections")
    def detections():
        return jsonify(session.latest_detections())

    @app.route("/api/camera/start", methods=["POST"])
    def camera_start():
        try:
            result = session.start_camera()
        except CameraError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 409
        return jsonify(result)

    @app.route("/api/camera/stop", methods=["POST"])
    def camera_stop():
        return jsonify(session.stop_camera())

    @app.route("/video_feed")
    def video_feed():
        if not session.camera_active:
            return jsonify({"status": "error", "message": "Camera is not started."}), 409

        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

        def generate():
            while session.camera_active:
                jpeg = session.latest_jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                yield boundary + jpeg + b"\r\n"
                time.sleep(0.04)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/events")
    def events():
        return jsonify({"status": "success", "summary": session.session_summary()})

    @app.route("/api/events/reset", methods=["POST"])
    def events_reset():
        session.reset_events()
        return jsonify({"status": "success", "summary": session.session_summary()})

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"status": "error", "message": "Endpoint not found."}), 404
        return render_template("index.html"), 404

    @app.errorhandler(Exception)
    def unhandled(error):
        if isinstance(error, ModelNotFoundError):
            return jsonify({"status": "model_missing", "message": str(error)}), 503
        app.logger.exception("Unhandled error on %s %s", request.method, request.path)
        return jsonify({"status": "error", "message": str(error)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    print(f"[server] starting on http://{config.HOST}:{config.PORT}")
    if not app.extensions["session"].model_loaded:
        print("[server] WARNING: no trained model found - detection endpoints will return 503.")
        print("[server] Train one with: python -m training.train --data-dir dataset")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
