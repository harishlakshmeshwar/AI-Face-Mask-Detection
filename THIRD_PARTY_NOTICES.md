# Third-Party Notices

This project is an original implementation. The components and references
listed below were used, consulted, or are redistributed as part of
dependencies. No substantial source code was copied from the reference
repositories listed in section 2.

## 1. Runtime dependencies (installed via pip)

| Component | License | Notes |
|---|---|---|
| Flask | BSD-3-Clause | Web server |
| Flask-CORS | MIT | CORS headers for the JSON API |
| OpenCV (`opencv-python`) | Apache-2.0 | Face detection and video I/O |
| TensorFlow / Keras (`tensorflow-cpu`) | Apache-2.0 | CNN training and inference |
| NumPy | BSD-3-Clause | Array operations |
| Pillow | MIT-CMU | Image file decoding (used by TF datasets) |
| scikit-learn | BSD-3-Clause | Precision/recall/F1 and confusion matrix |
| python-dotenv | BSD-3-Clause | `.env` loading |
| waitress | ZPL-2.1 | Production WSGI server on Windows |
| pytest | MIT | Test runner |

Exact installed versions are resolved by pip at install time; see
`requirements.txt` for the supported ranges.

## 2. OpenCV Haar cascade

`src/detector.py` loads `haarcascade_frontalface_default.xml` from the
`cv2.data.haarcascades` directory shipped inside the `opencv-python` wheel.
That XML file is part of the OpenCV project (Apache-2.0) and is not
vendored into this repository.

## 3. Reference repositories (consulted, not copied)

These public repositories were studied for general implementation patterns
(CNN classification flow, Haar-cascade detection, OpenCV frame handling).
Their code was **not** copied into this project; the architecture, naming,
API surface, and implementation here are original.

1. **jaybfn/Face-Mask-Detection-using-TensorFlow-and-OpenCV**
   - URL: https://github.com/jaybfn/Face-Mask-Detection-using-TensorFlow-and-OpenCV
   - License: MIT
   - Consulted for: high-level CNN + OpenCV workflow expectations.

2. **salar-shdk/face-mask-detection**
   - URL: https://github.com/salar-shdk/face-mask-detection
   - License: GPL-3.0
   - Consulted for: dataset folder-layout conventions and model-evaluation
     reporting patterns only. Because GPL-3.0 is a copyleft license, **no
     source code from this repository was reused**, and this project is not
     derived from it. Only general, non-copyrightable ideas (folder layout,
     metric reporting) were noted.

## 4. Datasets

This repository ships **no dataset**. Datasets are large, individually
licensed artifacts that users must obtain themselves. See the README
("Dataset preparation") for suggested sources and check each source's
license before use.

## 5. Model weights

No pretrained model weights are committed. The `models/` directory
contains only a `.gitkeep` placeholder. Train locally with
`python -m training.train --data-dir dataset` to generate
`models/mask_classifier.keras` and `models/class_labels.json`.
