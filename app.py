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
from typing import Any, Dict, List, Optional

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
SESSION_EVENT_PREVIEW = 50
JPEG_QUALITY = 80


def bbox_iou(box_a, box_b) -> float:
    """Intersection-over-union of two ``[x, y, w, h]`` boxes."""
    ax, ay, aw, ah = (float(v) for v in box_a)
    bx, by, bw, bh = (float(v) for v in box_b)
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class DetectionTracker:
    """Counts distinct face detections across consecutive video frames.

    Without this, every processed frame would add its faces to the session
    totals, so one person sitting in front of the camera would inflate
    "Faces seen" by the frame rate. A face is counted:

    * once when it first appears (IoU-matched to an existing track),
    * again if its classification flips (mask -> no mask or back),
    * again after it has left the view for more than ``MAX_MISSED_FRAMES``.

    Totals always satisfy ``with_mask + without_mask == faces``, which keeps
    the mask ratio (mask detections / total face detections) well defined.
    """

    IOU_THRESHOLD = 0.25
    MAX_MISSED_FRAMES = 3

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._tracks: List[Dict[str, Any]] = []
        self._next_id = 1
        self.frames = 0
        self.faces = 0
        self.with_mask = 0
        self.without_mask = 0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process one analysed frame; return newly counted detection events."""
        self.frames += 1
        events: List[Dict[str, Any]] = []

        # Greedy IoU matching between existing tracks and this frame's faces.
        pairs = []
        for track_index, track in enumerate(self._tracks):
            for det_index, detection in enumerate(detections):
                score = bbox_iou(track["bbox"], detection["bbox"])
                if score >= self.IOU_THRESHOLD:
                    pairs.append((score, track_index, det_index))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

        matched_tracks = set()
        matched_dets = set()
        preexisting = len(self._tracks)
        for _score, track_index, det_index in pairs:
            if track_index in matched_tracks or det_index in matched_dets:
                continue
            matched_tracks.add(track_index)
            matched_dets.add(det_index)
            track = self._tracks[track_index]
            detection = detections[det_index]
            track["missed"] = 0
            if detection.get("label") != track["label"]:
                # Same face, new classification: count it as a detection event.
                track["label"] = detection.get("label")
                events.append(self._count(detection, track["id"]))
            track["bbox"] = list(detection["bbox"])

        # Faces never seen before start a new track (and a new count).
        for det_index, detection in enumerate(detections):
            if det_index in matched_dets:
                continue
            track = {
                "id": self._next_id,
                "bbox": list(detection["bbox"]),
                "label": detection.get("label"),
                "missed": 0,
            }
            self._next_id += 1
            self._tracks.append(track)
            events.append(self._count(detection, track["id"]))

        # Age tracks that were not matched this frame; drop long-gone faces.
        for track_index in range(preexisting):
            if track_index not in matched_tracks:
                self._tracks[track_index]["missed"] += 1
        self._tracks = [
            track for track in self._tracks if track["missed"] <= self.MAX_MISSED_FRAMES
        ]
        return events

    def _count(self, detection: Dict[str, Any], track_id: int) -> Dict[str, Any]:
        self.faces += 1
        if detection.get("label") == "with_mask":
            self.with_mask += 1
        else:
            self.without_mask += 1
        return {
            "track_id": track_id,
            "label": detection.get("label"),
            "display_label": detection.get("display_label", detection.get("label")),
            "confidence": detection.get("confidence"),
            "bbox": list(detection.get("bbox", [])),
        }


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
        self.tracker = DetectionTracker()
        self._stats_lock = threading.Lock()
        self.events = deque(maxlen=MAX_EVENT_HISTORY)

    # ----------------------------------------------------------- model state
    @property
    def model_loaded(self) -> bool:
        return self.classifier.is_loaded

    # ------------------------------------------------------------- statistics
    def track(
        self, detections: List[Dict[str, Any]], latency_ms: float = 0.0, source: str = "api"
    ) -> None:
        """Account for one analysed frame.

        The tracker decides which detections are *new* (first appearance or a
        flipped classification); only those become counted events, so
        consecutive frames showing the same face never double-count.
        """
        with self._stats_lock:
            new_events = self.tracker.update(detections)
            if not new_events:
                return
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            for event in new_events:
                event["time"] = stamp[11:]  # HH:MM:SS for the detection log
                event["timestamp"] = stamp
                event["source"] = source
                event["latency_ms"] = latency_ms
                self.events.appendleft(event)

    def session_snapshot(self) -> Dict[str, Any]:
        """Session-wide statistics for the dashboard (and API consumers)."""
        with self._stats_lock:
            tracker = self.tracker
            return {
                "frames": tracker.frames,
                "faces": tracker.faces,
                "mask": tracker.with_mask,
                "no_mask": tracker.without_mask,
                # None (not 0) when nothing was seen: callers must not divide.
                "ratio": round(tracker.with_mask / tracker.faces, 4)
                if tracker.faces
                else None,
                "events": list(self.events)[:SESSION_EVENT_PREVIEW],
            }

    def session_summary(self) -> Dict[str, Any]:
        snapshot = self.session_snapshot()
        return {
            "frames": snapshot["frames"],
            "total_detections": snapshot["faces"],
            "with_mask": snapshot["mask"],
            "without_mask": snapshot["no_mask"],
            "mask_ratio": snapshot["ratio"],
            "events": snapshot["events"],
        }

    def reset_events(self) -> None:
        with self._stats_lock:
            self.tracker.reset()
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
        failures = 0
        max_failures = 8  # ~4 s of failed reads before giving up
        while not self._stop_event.is_set():
            started = time.time()
            ok, frame = self.camera.read()
            if not ok or frame is None:
                failures += 1
                self.camera_error = (
                    "Camera stopped delivering frames. It may have been disconnected "
                    "or claimed by another application (browser tab, Teams, Zoom)."
                )
                if failures >= max_failures:
                    self.camera_error = (
                        "Camera lost: no frames received for several seconds. "
                        "Close other applications using the webcam and restart detection."
                    )
                    self.camera_active = False
                    self.camera.stop()
                    self._stop_event.set()
                    break
                time.sleep(0.5)
                continue
            failures = 0

            if not self.classifier.is_loaded:
                # Preview-only: serve the raw frame so the operator can still
                # verify the camera, but never invent detections.
                with self._lock:
                    self._latest_jpeg = encode_jpeg(frame)
                    self._last_detections = {
                        "faces": [],
                        "counts": {"faces": 0, "with_mask": 0, "without_mask": 0},
                        "latency_ms": 0.0,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
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
            self.track(detections, latency, source="server_camera")

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
        payload["session"] = self.session_snapshot()
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
            "camera_error": session.camera_error,
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

        session.track(detections, latency, source="browser_camera")
        return jsonify(
            {
                "status": "success",
                "faces": detections,
                "counts": counts,
                "latency_ms": latency,
                "session": session.session_snapshot(),
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
