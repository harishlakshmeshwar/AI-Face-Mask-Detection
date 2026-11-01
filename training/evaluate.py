"""Evaluate a saved model on a folder-based dataset (for example a held-out set).

Usage::

    python -m training.evaluate --data-dir dataset/holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.inference import MaskClassifier, ModelNotFoundError
from src.preprocessing import to_model_input
from training.dataset import DatasetError, dataset_summary, validate_dataset_dir


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained face-mask model.")
    parser.add_argument("--data-dir", default=str(config.DATASET_DIR))
    parser.add_argument("--model-path", default=str(config.MODEL_PATH))
    parser.add_argument("--labels-path", default=str(config.CLASS_LABELS_PATH))
    parser.add_argument("--report-path", default=str(Path(config.MODELS_DIR) / "evaluation_report.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def evaluate(classifier: MaskClassifier, per_class, class_names: List[str], batch_size: int) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    images: List[np.ndarray] = []
    labels: List[int] = []
    for index, name in enumerate(class_names):
        for path in per_class.get(name, []):
            image = None
            try:
                import cv2

                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            except Exception:
                image = None
            if image is None:
                continue
            images.append(to_model_input(image, classifier.image_size))
            labels.append(index)

    if not images:
        raise DatasetError("No readable images found for evaluation.")

    batch = np.stack(images, axis=0)
    probs = np.asarray(classifier.model.predict(batch, verbose=0))
    predictions = np.argmax(probs, axis=1).tolist()

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "num_samples": len(labels),
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    print("=" * 64)
    print("  AI FACE MASK DETECTION - MODEL EVALUATION")
    print("=" * 64)

    try:
        per_class = validate_dataset_dir(args.data_dir)
    except DatasetError as exc:
        print(f"[error] {exc}")
        return 1

    print(f"[dataset] {dataset_summary(per_class)} from {args.data_dir}")

    try:
        classifier = MaskClassifier(model_path=args.model_path, labels_path=args.labels_path)
    except ModelNotFoundError as exc:
        print(f"[error] {exc}")
        return 1

    if not classifier.is_loaded:
        print(f"[error] {classifier.load_error}")
        return 1

    print(f"[model] loaded from {args.model_path}")
    print(f"[model] classes={classifier.class_names} image_size={classifier.image_size}")

    results = evaluate(classifier, per_class, classifier.class_names, args.batch_size)

    print("\n[results]")
    print(f"  samples : {results['num_samples']}")
    print(f"  accuracy: {results['accuracy']:.4f}")
    report = results["classification_report"]
    for name in classifier.class_names:
        row = report[name]
        print(
            f"  {name:<13} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1-score']:.3f}"
        )
    print("  confusion matrix (rows=true, cols=pred):")
    for row in results["confusion_matrix"]:
        print(f"    {row}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path),
        "classes": classifier.class_names,
        **results,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n[saved] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
