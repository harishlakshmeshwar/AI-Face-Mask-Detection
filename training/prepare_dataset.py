"""Prepare a folder-based dataset of face images for mask classification.

Many public "face mask" datasets ship scene images (a face somewhere in a
128-180 px frame) rather than tight face crops. This app classifies the
faces its detector finds, so training data should look the same: one face
per cropped image.

This script takes a source directory of class folders and writes a cropped
binary dataset (``with_mask`` / ``without_mask``) ready for
``python -m training.train``::

    python -m training.prepare_dataset --source /path/to/raw --dest dataset

Face crops keep 20 % margin around the detected box. Images where no face is
detected fall back to a centre crop (most public datasets are face-centred);
use ``--strict`` to drop them instead. Third class folders such as
``mask_weared_incorrect`` are ignored unless listed via ``--classes``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.detector import FaceDetector

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Crop face images into a binary mask dataset.")
    parser.add_argument("--source", required=True, help="Source directory containing class folders")
    parser.add_argument("--dest", default=str(config.DATASET_DIR), help="Destination dataset root")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["with_mask", "without_mask"],
        help="Source class folder names, in the order of the target binary classes",
    )
    parser.add_argument("--margin", type=float, default=0.2, help="Margin around the face box (fraction of size)")
    parser.add_argument("--strict", action="store_true", help="Skip images where no face is detected")
    parser.add_argument("--min-neighbors", type=int, default=3, help="Haar detectMultiScale minNeighbors")
    return parser.parse_args(argv)


def face_crop(image: np.ndarray, detector: FaceDetector, margin: float, strict: bool, min_neighbors: int):
    """Return the largest detected face crop, a centre-crop fallback, or None."""
    faces = detector.detect(image)
    # Re-run with relaxed parameters: masked faces often need fewer neighbours.
    if not faces:
        relaxed = detector._cascade.detectMultiScale(
            cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
            scaleFactor=1.08,
            minNeighbors=min_neighbors,
            minSize=(20, 20),
        )
        faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in relaxed]

    h_img, w_img = image.shape[:2]

    if faces:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(x - mx, 0), max(y - my, 0)
        x1, y1 = min(x + w + mx, w_img), min(y + h + my, h_img)
        crop = image[y0:y1, x0:x1]
        if crop.size > 0:
            return crop, "face"

    if strict:
        return None, "skipped"

    side = int(min(h_img, w_img) * 0.75)
    cx, cy = (w_img - side) // 2, (h_img - side) // 2
    return image[cy : cy + side, cx : cx + side], "center"


def main(argv=None) -> int:
    args = parse_args(argv)
    source = Path(args.source)
    dest = Path(args.dest)

    if not source.is_dir():
        print(f"[error] Source directory '{source}' does not exist.")
        return 1
    if len(args.classes) != 2:
        print("[error] Exactly two source class names are required (binary dataset).")
        return 1

    detector = FaceDetector()
    dest.mkdir(parents=True, exist_ok=True)

    totals = {}
    for source_class in args.classes:
        class_dir = source / source_class
        if not class_dir.is_dir():
            # tolerate a single nested folder level (e.g. dataset/data/with_mask)
            nested = list(source.glob(f"*/{source_class}"))
            if nested:
                class_dir = nested[0]
            else:
                print(f"[error] Missing class folder: {class_dir}")
                return 1

        target_dir = dest / source_class
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        saved = fallback = skipped = 0
        for index, path in enumerate(images):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                skipped += 1
                continue
            crop, kind = face_crop(image, detector, args.margin, args.strict, args.min_neighbors)
            if crop is None:
                skipped += 1
                continue
            out = target_dir / f"{source_class}_{index:05d}.jpg"
            if cv2.imwrite(str(out), crop):
                saved += 1
                if kind == "center":
                    fallback += 1
            else:
                skipped += 1

        totals[source_class] = saved
        print(f"[{source_class}] source={len(images)} saved={saved} center_fallback={fallback} skipped={skipped}")

    total = sum(totals.values())
    print(f"[done] {total} cropped images written to '{dest}'")
    if total == 0:
        print("[error] No images were produced - check the source directory.")
        return 1
    print("[next] python -m training.train --data-dir", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
