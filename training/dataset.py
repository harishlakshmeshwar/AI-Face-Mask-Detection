"""Dataset discovery, validation and deterministic splitting."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

EXPECTED_CLASSES = ("with_mask", "without_mask")


class DatasetError(ValueError):
    """Raised when the dataset directory is missing or malformed."""


def list_image_paths(class_dir: Path) -> List[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset_dir(data_dir: str | Path) -> Dict[str, List[Path]]:
    """Validate the folder-based dataset layout and return paths per class.

    Expected layout::

        dataset/
        ├── with_mask/
        │   ├── img001.jpg
        │   └── ...
        └── without_mask/
            ├── img001.jpg
            └── ...
    """
    root = Path(data_dir)
    if not root.exists():
        raise DatasetError(
            f"Dataset directory '{root}' does not exist. "
            "Download a face-mask dataset and place it at this path "
            "(see README 'Dataset preparation')."
        )
    if not root.is_dir():
        raise DatasetError(f"Dataset path '{root}' is not a directory.")

    per_class: Dict[str, List[Path]] = {}
    missing = [name for name in EXPECTED_CLASSES if not (root / name).is_dir()]
    if missing:
        raise DatasetError(
            f"Dataset directory '{root}' is missing class folder(s): {', '.join(missing)}. "
            f"Expected sub-folders named {', '.join(EXPECTED_CLASSES)}."
        )

    for name in EXPECTED_CLASSES:
        images = list_image_paths(root / name)
        if not images:
            raise DatasetError(
                f"Class folder '{root / name}' contains no supported image files "
                f"({', '.join(IMAGE_EXTENSIONS)})."
            )
        per_class[name] = images

    return per_class


def dataset_summary(per_class: Dict[str, List[Path]]) -> Dict[str, int]:
    summary = {name: len(paths) for name, paths in per_class.items()}
    summary["total"] = sum(summary.values())
    return summary


def split_dataset(
    per_class: Dict[str, List[Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """Split each class into train/validation lists without overlap."""
    if not 0.0 < val_ratio < 1.0:
        raise DatasetError("val_ratio must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)
    train: Dict[str, List[Path]] = {}
    val: Dict[str, List[Path]] = {}

    for name, paths in per_class.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        val[name] = sorted(shuffled[:n_val])
        train[name] = sorted(shuffled[n_val:])

    return train, val


def check_split_overlap(train: Dict[str, List[Path]], val: Dict[str, List[Path]]) -> None:
    """Fail loudly if any file appears in both splits."""
    for name in train:
        overlap = set(train[name]) & set(val.get(name, []))
        if overlap:
            raise DatasetError(
                f"Train/validation overlap for class '{name}': {len(overlap)} shared file(s)."
            )
