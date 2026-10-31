"""Central configuration resolved from environment variables and project paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))
DATASET_DIR = Path(os.environ.get("DATASET_DIR", BASE_DIR / "dataset"))

MODEL_PATH = Path(os.environ.get("MODEL_PATH", MODELS_DIR / "mask_classifier.keras"))
CLASS_LABELS_PATH = Path(os.environ.get("CLASS_LABELS_PATH", MODELS_DIR / "class_labels.json"))

DEFAULT_IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "128"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.50"))

CAMERA_SOURCE_RAW = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(CAMERA_SOURCE_RAW) if CAMERA_SOURCE_RAW.isdigit() else CAMERA_SOURCE_RAW

PORT = int(os.environ.get("PORT", "5000"))
HOST = os.environ.get("HOST", "127.0.0.1")
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

CLASS_NAMES = ["with_mask", "without_mask"]
DISPLAY_NAMES = {"with_mask": "Mask", "without_mask": "No Mask"}

EVENT_HISTORY_LIMIT = int(os.environ.get("EVENT_HISTORY_LIMIT", "200"))
CAMERA_FPS = float(os.environ.get("CAMERA_FPS", "15"))
