# Model Card: Face-Mask CNN Classifier

Last updated: pending first training run.

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
- **Provenance:** user-supplied. This repository does not bundle a dataset.
  Suggested public sources are listed in the README; each user must verify
  the license of whatever dataset they download.
- **Size / class balance:** pending (whatever the user provides; the trainer
  prints a per-class image count at startup).
- **Split:** deterministic per-class train/validation split
  (`--val-split`, default 20%, `--seed`, default 42). Paths are checked for
  overlap so no image appears in both sets.

## Evaluation methodology

- Held-out validation split from the same dataset (never trained on).
- Metrics written to `models/metrics.json` after training:
  - validation loss and accuracy
  - per-class precision, recall, and F1 (`sklearn.metrics.classification_report`)
  - confusion matrix (rows = true class, cols = predicted class)
- Additional offline evaluation against any folder-based set:
  `python -m training.evaluate --data-dir <folder>` writes
  `models/evaluation_report.json`.
- **Measured results:** pending — no training run has been recorded in this
  repository yet. Do not treat any accuracy figure as known until a real
  training run produces `models/metrics.json`.

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
