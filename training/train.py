"""Train a compact CNN that classifies face crops as Mask / No Mask.

Usage::

    python -m training.train --data-dir dataset --epochs 10

The script validates the dataset, trains with augmentation, evaluates on a
held-out split, writes a confusion matrix plus precision/recall/F1 scores,
and saves the model together with the class-label configuration required by
the inference code. Nothing is claimed about accuracy unless training runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from training.dataset import (
    DatasetError,
    check_split_overlap,
    dataset_summary,
    split_dataset,
    validate_dataset_dir,
)

CLASS_NAMES: List[str] = list(config.CLASS_NAMES)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face-mask CNN classifier.")
    parser.add_argument("--data-dir", default=str(config.DATASET_DIR), help="Dataset root with with_mask/ and without_mask/")
    parser.add_argument("--model-dir", default=str(config.MODELS_DIR), help="Directory for saved model and metrics")
    parser.add_argument("--img-size", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augmentation", action="store_true", help="Disable training-time augmentation")
    parser.add_argument("--dropout", type=float, default=0.4)
    return parser.parse_args(argv)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception as exc:
        print(f"[warn] Could not set TensorFlow seed: {exc}")


def build_datasets(
    per_class: Dict[str, List[Path]],
    image_size: int,
    batch_size: int,
    seed: int,
    augment: bool,
):
    """Build tf.data pipelines for train and validation splits."""
    import tensorflow as tf

    def paths_and_labels(split: Dict[str, List[Path]]) -> Tuple[List[str], List[int]]:
        files: List[str] = []
        labels: List[int] = []
        for index, name in enumerate(CLASS_NAMES):
            for path in split.get(name, []):
                files.append(str(path))
                labels.append(index)
        return files, labels

    def decode(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.cast(image, tf.float32) / 255.0  # RGB, [0, 1] - same as inference
        return image, label

    train_files, train_labels = paths_and_labels(per_class["train"])
    val_files, val_labels = paths_and_labels(per_class["val"])

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        .shuffle(len(train_files), seed=seed)
        .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    )
    if augment:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomContrast(0.10),
            ],
            name="augmentation",
        )
        train_ds = train_ds.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = None
    if val_files:
        val_ds = (
            tf.data.Dataset.from_tensor_slices((val_files, val_labels))
            .map(decode, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    return train_ds, val_ds, len(train_files), len(val_files)


def build_model(image_size: int, num_classes: int, dropout: float):
    """Compact CNN suited to local demos and CPU training."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3), name="input"),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(max(dropout - 0.1, 0.1)),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="mask_classifier_cnn",
    )
    return model


def evaluate_split(model, val_ds, class_names: List[str]) -> Dict:
    """Compute accuracy plus per-class precision/recall/F1 and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    y_true: List[int] = []
    y_pred: List[int] = []
    for batch_images, batch_labels in val_ds:
        probs = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.asarray(batch_labels).tolist())

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "num_samples": len(y_true),
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_global_seed(args.seed)

    print("=" * 64)
    print("  AI FACE MASK DETECTION - CNN TRAINING")
    print("=" * 64)

    try:
        per_class_images = validate_dataset_dir(args.data_dir)
    except DatasetError as exc:
        print(f"[error] {exc}")
        return 1

    summary = dataset_summary(per_class_images)
    print(f"[dataset] {summary}")
    print(f"[dataset] classes: {', '.join(CLASS_NAMES)}")

    train_split, val_split = split_dataset(per_class_images, val_ratio=args.val_split, seed=args.seed)
    check_split_overlap(train_split, val_split)
    train_summary = dataset_summary(train_split)
    val_summary = dataset_summary(val_split)
    print(f"[split] train={train_summary}  val={val_summary}  seed={args.seed}")
    if val_summary["total"] == 0:
        print("[warn] Validation split is empty; metrics will be limited.")

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[error] TensorFlow is not available: {exc}")
        print("Install dependencies first: pip install -r requirements.txt")
        return 1

    train_ds, val_ds, n_train, n_val = build_datasets(
        {"train": train_split, "val": val_split},
        image_size=args.img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        augment=not args.no_augmentation,
    )

    model = build_model(args.img_size, len(CLASS_NAMES), dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    model.summary()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "mask_classifier.keras"
    labels_path = model_dir / "class_labels.json"
    history_path = model_dir / "training_history.csv"
    metrics_path = model_dir / "metrics.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy" if n_val else "accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if n_val else "accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if n_val else "loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path)),
    ]

    print(f"[train] images={n_train} val_images={n_val} img_size={args.img_size} batch={args.batch_size} epochs={args.epochs} lr={args.lr} augmentation={'off' if args.no_augmentation else 'on'}")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    # Ensure the final (best-restored) weights are on disk even if the last
    # epoch was not the checkpointed best.
    model.save(model_path)
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump({"labels": CLASS_NAMES, "image_size": args.img_size}, handle, indent=2)

    metrics: Dict = {
        "image_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_run": len(history.history.get("loss", [])),
        "seed": args.seed,
        "train_samples": n_train,
        "val_samples": n_val,
        "train_split": {k: len(v) for k, v in train_split.items()},
        "val_split": {k: len(v) for k, v in val_split.items()},
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_training_loss": float(history.history["loss"][-1]),
    }

    if val_ds is not None and n_val > 0:
        val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
        metrics["val_loss"] = float(val_loss)
        metrics["val_accuracy"] = float(val_accuracy)
        metrics.update(evaluate_split(model, val_ds, CLASS_NAMES))

        print("\n[evaluation]")
        print(f"  val_loss     : {val_loss:.4f}")
        print(f"  val_accuracy : {val_accuracy:.4f}")
        report = metrics["classification_report"]
        for name in CLASS_NAMES:
            row = report[name]
            print(
                f"  {name:<13} precision={row['precision']:.3f} "
                f"recall={row['recall']:.3f} f1={row['f1-score']:.3f}"
            )
        print(f"  confusion matrix (rows=true, cols=pred) {CLASS_NAMES}:")
        for row in metrics["confusion_matrix"]:
            print(f"    {row}")
    else:
        print("[warn] Skipping validation metrics - no validation images.")

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n[saved] {model_path}")
    print(f"[saved] {labels_path}")
    print(f"[saved] {history_path}")
    print(f"[saved] {metrics_path}")
    print("[done] Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
