# AI Face Mask Detection System

Real-time face-mask detection with **Python, OpenCV and a TensorFlow/Keras CNN**.
Faces are detected in a camera stream, each face crop is classified as
`Mask` or `No Mask`, and the result is drawn back onto the frame with a
confidence score.

> Built a real-time face-mask detection application using Python, OpenCV and
> a CNN model with frame-by-frame inference and visual alerts. Developed a
> TensorFlow/Keras training and evaluation pipeline for mask classification,
> with reproducible preprocessing and model deployment instructions.

> **Repository note:** this project lives in the existing GitHub repository
> `NEXUS-AI-FaceRecognition`. The code, README and structure now implement the
> face-mask system; the repository name itself was not changed.

---

## Table of contents

- [Key features](#key-features)
- [Technology stack](#technology-stack)
- [Architecture and workflow](#architecture-and-workflow)
- [Repository structure](#repository-structure)
- [System requirements](#system-requirements)
- [Installation (Windows)](#installation-windows)
- [Dataset preparation](#dataset-preparation)
- [Training and evaluation](#training-and-evaluation)
- [Model file setup](#model-file-setup)
- [Running the application](#running-the-application)
- [Using the web interface](#using-the-web-interface)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citations and licences](#citations-and-licences)
- [Limitations](#limitations)
- [Screenshots](#screenshots)

---

## Key features

- **Two detection modes**
  - *Browser camera* (recommended): the web page captures frames with
    `getUserMedia` and posts them to `POST /api/detect`; boxes are drawn on a
    canvas. Works even when the Python process cannot open the webcam.
  - *Server webcam (OpenCV)*: `POST /api/camera/start` opens the webcam from
    Python, a background thread runs detection, and `GET /video_feed` serves
    an annotated MJPEG stream. Camera resources are released on
    `POST /api/camera/stop` and at process exit.
- **Per-face classification** — never classifies a whole frame as one face;
  every detected box is cropped, preprocessed and run through the CNN.
- **Bounding boxes + confidence** — green for `Mask`, red for `No Mask`,
  with the confidence percentage on each box.
- **Dashboard** — live counts of faces / masks / no-masks, per-face
  confidence bars, session totals, mask ratio, and a detection log.
- **Clear states** — model missing, camera denied, camera unavailable and
  backend offline are all surfaced in the UI.
- **Reproducible training** — fixed seed, deterministic split with overlap
  check, training-only augmentation, confusion matrix, per-class
  precision/recall/F1, saved history and metrics JSON.
- **Responsive, accessible UI** — keyboard focus styles, skip link, live
  regions, reduced-motion support, no external CDN dependencies.

## Technology stack

| Layer | Choice |
|---|---|
| Backend | Python 3.10+, Flask, Waitress (production WSGI) |
| Vision | OpenCV (`opencv-python`) — Haar cascade face detection |
| ML | TensorFlow / Keras CNN, scikit-learn metrics |
| Frontend | Vanilla HTML/CSS/JS (no framework, no build step) |
| Tests | pytest with a mocked model (no webcam needed) |

## Architecture and workflow

```
                       ┌──────────────────────────────────────────┐
  Browser camera mode  │  Browser (getUserMedia)                 │
  ───────────────────  │   video → canvas → JPEG data-URL        │
                       └───────────────┬──────────────────────────┘
                                       │ POST /api/detect
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Flask (app.py)                                                          │
│                                                                          │
│   ┌──────────────┐    ┌────────────────┐    ┌───────────────────────┐    │
│   │ FaceDetector │───▶│ preprocess     │───▶│ MaskClassifier (CNN)  │    │
│   │ (Haar XML)   │    │ BGR→RGB, size, │    │ models/*.keras        │    │
│   │              │    │ [0,1] float32  │    │ class_labels.json     │    │
│   └──────────────┘    └────────────────┘    └───────────┬───────────┘    │
│                                                         │                │
│   Server webcam mode: WebcamCamera → worker thread ─────┘                │
│                       (annotated JPEG → GET /video_feed)                 │
└──────────────────────────────────────────────────────────────────────────┘
                                       │ JSON: faces[], counts, latency_ms
                                       ▼
                       Dashboard (canvas overlay / MJPEG <img>)
```

**Per-frame pipeline:** frame → Haar face detection → crop each face →
`BGR→RGB`, resize to the training image size, scale to `[0,1]` → CNN
softmax → label + confidence → draw box → return JSON / stream.

The preprocessing contract is shared between training and inference in
`src/preprocessing.py`, so the model always sees the same kind of tensor.

## Repository structure

```
.
├── app.py                  # Flask app: /api/detect, /api/camera/*, /video_feed, UI
├── requirements.txt
├── run_local.bat           # Windows quick-start
├── .env.example
├── README.md
├── THIRD_PARTY_NOTICES.md
├── src/
│   ├── config.py           # paths, env vars, class names
│   ├── preprocessing.py    # BGR→RGB, resize, normalize (shared contract)
│   ├── detector.py         # OpenCV Haar cascade wrapper
│   ├── inference.py        # MaskClassifier: load, predict, annotate
│   └── camera.py           # WebcamCamera with guaranteed release
├── training/
│   ├── dataset.py          # dataset validation + deterministic split
│   ├── train.py            # python -m training.train
│   └── evaluate.py         # python -m training.evaluate
├── templates/index.html    # dashboard
├── static/
│   ├── css/style.css
│   └── js/app.js
├── models/                 # generated artifacts (git-ignored, .gitkeep only)
├── tests/                  # pytest suite (mocked model, no webcam)
└── docs/model_card.md
```

## System requirements

- Windows 10/11 (macOS and Linux also work; commands below note differences)
- Python **3.10 – 3.12** (3.12 tested in development)
- A webcam for live use (optional for API/tests)
- ~1.5 GB free disk for TensorFlow + dependencies

## Installation (Windows)

```bat
git clone https://github.com/harishlakshmeshwar/NEXUS-AI-FaceRecognition.git
cd NEXUS-AI-FaceRecognition

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS equivalents:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and adjust `PORT`, `CAMERA_SOURCE`
(camera index for server-webcam mode) or `CONFIDENCE_THRESHOLD`.

> Windows note: `tensorflow-cpu` in `requirements.txt` is the CPU build,
> which is sufficient for training and inference here. If you have a
> supported NVIDIA GPU setup you can install `tensorflow` instead.

## Dataset preparation

The trainer expects a simple folder-based dataset:

```
dataset/
├── with_mask/
│   ├── img_0001.jpg
│   └── ...
└── without_mask/
    ├── img_0001.jpg
    └── ...
```

**Where to get one (verify the licence of every source yourself):**

- Kaggle — search for *"Face Mask Detection Dataset"* (several public
  datasets provide `with_mask` / `without_mask` folders directly).
- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net) — synthetic
  masked faces built on FFHQ; check its research-use terms.
- [FFHQ](https://github.com/NVlabs/ffhq-dataset) — high-quality faces you
  can pair with masked images; licensed separately.

**Already have scene images instead of face crops?** Many public datasets
ship whole frames (a face somewhere in a 128-180 px image). This app
classifies detected face crops, so convert them first:

```bat
python -m training.prepare_dataset --source C:\path\to\raw_dataset --dest dataset
```

The script detects the face in each image, keeps a 20 % margin crop, and
writes a balanced binary `dataset/` ready for training (see
`training/prepare_dataset.py --help` for options such as `--strict`).

Place the extracted folders so the two class directories sit directly under
`dataset/`, then sanity-check with:

```bat
python -c "from training.dataset import validate_dataset_dir, dataset_summary; print(dataset_summary(validate_dataset_dir('dataset')))"
```

Datasets are git-ignored; nothing large is ever committed.

## Training and evaluation

```bat
REM Train (defaults shown; use --help for everything)
python -m training.train --data-dir dataset --img-size 128 --batch-size 32 --epochs 10 --lr 0.001 --val-split 0.2 --seed 42

REM Evaluate the saved model on any folder-based set (e.g. a held-out split)
python -m training.evaluate --data-dir dataset
```

Training writes:

| File | Contents |
|---|---|
| `models/mask_classifier.keras` | trained model |
| `models/class_labels.json` | class order + image size used at inference |
| `models/training_history.csv` | per-epoch loss/accuracy |
| `models/metrics.json` | val loss/accuracy, confusion matrix, per-class P/R/F1 |
| `models/evaluation_report.json` | produced by `training.evaluate` |

Early stopping restores the best weights; a fixed seed controls Python,
NumPy and TensorFlow.

**No trained model ships with this repository.** Until you run training, the
web UI shows a "model missing" banner and `POST /api/detect` returns `503`.
This is intentional — no dummy weights or invented accuracy numbers are
included.

## Model file setup

1. Obtain/prepare the dataset (above).
2. `python -m training.train --data-dir dataset`
3. Confirm `models/mask_classifier.keras` and `models/class_labels.json`
   exist.
4. Start the app — the header pill should read **Model: loaded**.

If the label file and model disagree (e.g. you swapped in a foreign
`.keras` file), retrain so both are regenerated together.

## Running the application

```bat
python app.py
REM or, for a production WSGI server:
waitress-serve --host=127.0.0.1 --port=5000 app:app
REM or double-click run_local.bat
```

Open **http://127.0.0.1:5000** in Chrome, Edge or Firefox.

### API surface

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/api/health` | backend/model/detector status |
| POST | `/api/detect` | classify one frame: `{"image": "<data-url>"}` |
| POST | `/api/camera/start` | open the server webcam (OpenCV) |
| POST | `/api/camera/stop` | release the webcam |
| GET | `/video_feed` | annotated MJPEG stream (server mode) |
| GET | `/api/detections` | latest server-mode detections + counts |
| GET | `/api/events` | session detection summary + recent events |
| POST | `/api/events/reset` | clear the server-side event log |

## Using the web interface

1. Choose **Browser camera** or **Server webcam** in the mode dropdown.
2. Click **Start detection** and allow camera access if prompted.
3. Watch the live preview: each face gets a coloured box with a label and
   confidence.
4. The right-hand panel shows current-frame counts, per-face confidence
   bars, session totals, mask ratio and a detection log.
5. **Stop** releases the camera (browser tracks are stopped; the server
   webcam is closed with `VideoCapture.release()`).
6. **Reset session** clears counters and the log.

Hosting note: a deployed web service cannot reach your laptop's webcam. If
you host the dashboard elsewhere, use **Browser camera** mode (frames travel
from your browser to the API) or run the whole stack locally.

## Testing

```bat
pytest -q
```

The suite uses a mocked model and synthetic frames, so it needs neither a
webcam nor trained weights. Covered: preprocessing shape/colour contract,
detector initialisation, prediction-label mapping, missing-model handling,
invalid/empty image inputs, camera start/stop API behaviour, and the health
and detect routes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Model file not found" banner | Train first: `python -m training.train --data-dir dataset` |
| `503 model_missing` from `/api/detect` | Same as above — the API refuses to pretend a prediction exists without weights |
| "Camera permission denied" | Allow camera access for `localhost` in the browser, or switch to server-webcam mode |
| "Camera source 0 is unavailable" | Close Zoom/Teams/other apps using the webcam, or set `CAMERA_SOURCE=1` in `.env` |
| `Dataset directory ... does not exist` | Place the class folders under `dataset/` as shown above |
| TensorFlow import errors on Windows | Use Python 3.10–3.12 and reinstall: `pip install --force-reinstall tensorflow-cpu` |
| Very low FPS | Reduce `--img-size` when training, close other GPU/CPU-heavy apps, or use browser mode |
| `Address already in use` | Another process owns the port: set `PORT=5001` in `.env` |

## Citations and licences

- This project's code: MIT License (see repository licence).
- Third-party dependencies and reference repositories: see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Datasets: not included. Respect each dataset's licence.
- Model architecture and evaluation details, limitations and failure cases:
  [`docs/model_card.md`](docs/model_card.md).

Reference projects consulted for general patterns (no code copied):
[jaybfn/Face-Mask-Detection-using-TensorFlow-and-OpenCV](https://github.com/jaybfn/Face-Mask-Detection-using-TensorFlow-and-OpenCV) (MIT) and
[salar-shdk/face-mask-detection](https://github.com/salar-shdk/face-mask-detection) (GPL-3.0).

## Limitations

- Haar-cascade detection misses profile/occluded/small faces; those faces
  are never classified.
- Binary classes only — no mask-type detection, no liveness/spoofing checks.
- Performance is dataset-dependent; no accuracy is claimed here because no
  training run has been recorded in this repository yet.
- CPU inference limits frame rate with multiple faces in view.
- Not a medical device; output is advisory only.

## Screenshots

> Placeholders only — replace with genuine captures after running the app
> with a trained model.
>
> - `docs/screenshots/dashboard-idle.png` *(pending)*
> - `docs/screenshots/detection-mask.png` *(pending)*
> - `docs/screenshots/detection-no-mask.png` *(pending)*
> - `docs/screenshots/model-missing-state.png` *(pending)*
