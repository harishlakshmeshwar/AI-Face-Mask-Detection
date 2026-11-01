# Model Card: Face-Mask CNN Classifier

Last updated: first training run recorded (local, CPU).

## Model details

- **Name:** `mask_classifier_cnn` (see `training/train.py:build_model`)
- **Task:** binary image classification of detected face crops
- **Classes:** `with_mask`, `without_mask` (fixed order in `src/config.py:CLASS_NAMES`)
- **Input:** RGB image, `(image_size, image_size, 3)`, float32 in `[0, 1]`
  (default `image_size = 128`, recorded in `models/class_labels.json` at training time)
- **Output:** softmax probability over the two classes
- **Framework:** TensorFlow / Keras 3
- **Approximate size:** < 1M parameters (compact CPU-friendly CNN with
  three Conv2D blocks, global average pooling, and dense head)
- **Artifact:** `models/mask_classifier.keras` (generated locally, not committed)

## Intended use

- Educational demonstrations, student portfolio projects, and technical
  interviews.
- Real-time screening support: highlight whether a detected face appears to
  be wearing a mask, for a human operator to review.

### Out-of-scope uses

- Not a medical device. Must not be used for diagnosis, enforcement, or any
  high-stakes decision about an individual.
- Not trained or validated for clinical populations, children, or any
  demographic group specifically.

## Training data

- **Expected layout:** `dataset/with_mask/` and `dataset/without_mask/`
  containing JPG/PNG/BMP/WebP images.
- **Provenance (run recorded here):** Kaggle *"Face Mask Detection"*
  (`vijaykumar1799/face-mask-detection`, mirrored on Hugging Face as
  `Kai1014/facemask-kaggle`), **ODbL-1.0**. 1,120 images after preparation
  (560 `with_mask` / 560 `without_mask`), balanced.
- **Preparation:** images were 128 px scene photos, so
  `training/prepare_dataset.py` face-cropped each one (Haar detection with
  20 % margin; 466/1,120 images had no detectable face and used a centre
  crop fallback). The `mask_weared_incorrect` class was excluded
  (binary task).
- **Split:** deterministic per-class train/validation split (20 %, seed 42),
  overlap-checked: 896 train / 224 validation.
- Datasets are not redistributed by this repository (`dataset/` is
  git-ignored); see `THIRD_PARTY_NOTICES.md` for attribution.

## Evaluation methodology

- Held-out validation split from the same dataset (never trained on).
- Metrics written to `models/metrics.json` after training:
  - validation loss and accuracy
  - per-class precision, recall, and F1 (`sklearn.metrics.classification_report`)
  - confusion matrix (rows = true class, cols = predicted class)
- Additional offline evaluation against any folder-based set:
  `python -m training.evaluate --data-dir <folder>` writes
  `models/evaluation_report.json`.
- **Measured results (this run, `models/metrics.json`):**
  - validation accuracy **0.857**, validation loss **0.337**
  - `with_mask`: precision 0.857, recall 0.857, F1 0.857
  - `without_mask`: precision 0.857, recall 0.857, F1 0.857
  - confusion matrix (rows = true, cols = predicted): `[[96, 16], [16, 96]]`
  - command: `python -m training.train --data-dir dataset --img-size 96
    --batch-size 32 --epochs 8 --lr 0.001 --val-split 0.2 --seed 42`
  - 8 epochs on CPU, best weights restored from epoch 6.

## Hyperparameters (defaults)

| Parameter | Default | Flag |
|---|---|---|
| Image size | 128 | `--img-size` |
| Batch size | 32 | `--batch-size` |
| Epochs | 10 (early stopping patience 4) | `--epochs` |
| Optimizer | Adam | fixed |
| Learning rate | 1e-3 (auto-reduced on plateau) | `--lr` |
| Dropout | 0.4 / 0.3 | `--dropout` |
| Augmentation | horizontal flip, small rotation/zoom/contrast (training only) | `--no-augmentation` to disable |
| Seed | 42 (Python, NumPy, TensorFlow) | `--seed` |

## Fairness and bias

- **Pending.** No demographic audit has been performed. Public mask datasets
  are often collected from limited sources and may under-represent some skin
  tones, ages, or regions. Expect uneven performance across groups and test
  with your own camera before drawing any conclusions.

## Limitations and known failure cases

- **Haar cascade detection is the bottleneck.** Profile views, heavy
  occlusion, very small faces, low light, and unusual angles cause missed
  detections, which means the classifier never sees those faces at all.
- **Surgical vs. cloth vs. N95:** the binary `with_mask` class does not
  distinguish mask types; a loosely worn or chin-only mask may be classified
  either way.
- **Spoofing:** the model classifies pixels, not people. Photos, screens, or
  printed masks can fool it. It provides no liveness detection.
- **Single-face crop assumption:** each detected box is classified
  independently; overlapping faces or reflections can produce misleading
  per-face labels.
- **Domain shift:** performance drops when the camera, lighting, or
  population differs substantially from the training data.
- **CPU latency:** inference runs on CPU by default; expect modest FPS with
  multiple faces in frame.

## Environmental impact

- Training runs locally on CPU by default; energy cost is small for a single
  short run but scales with dataset size and epoch count.

## Contact / maintenance

Maintained in the project repository. Report issues with a description of the
dataset used, the command run, and the contents of `models/metrics.json`.
