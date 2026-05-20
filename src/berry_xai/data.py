from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PAPER_CLASS_ORDER = (
    "blackberry",
    "blueberry",
    "tayberry",
    "white mulberry",
    "wineberry",
)

CLASS_NAME_ALIASES = {
    "black berry": "blackberry",
    "blackberry": "blackberry",
    "blueberry": "blueberry",
    "tayberry": "tayberry",
    "white mulberry": "white mulberry",
    "winberry": "wineberry",
    "wineberry": "wineberry",
}

DISPLAY_CLASS_NAMES = {
    "blackberry": "Blackberry",
    "blueberry": "Blueberry",
    "tayberry": "Tayberry",
    "white mulberry": "White Mulberry",
    "wineberry": "Wineberry",
}


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int
    class_name: str


@dataclass(frozen=True)
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    train_records: tuple[SampleRecord, ...]
    val_records: tuple[SampleRecord, ...]
    test_records: tuple[SampleRecord, ...]
    class_names: tuple[str, ...]
    class_weights: dict[int, float]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class GeneratorBundle:
    train_generator: object
    val_generator: object
    class_names: tuple[str, ...]
    class_weights: dict[int, float]
    image_size: tuple[int, int]
    validation_split: float
    dataset_root: Path


def normalize_class_name(name: str) -> str:
    normalized = " ".join(name.strip().lower().replace("_", " ").split())
    return CLASS_NAME_ALIASES.get(normalized, normalized)


def display_class_name(name: str) -> str:
    return DISPLAY_CLASS_NAMES.get(normalize_class_name(name), name.title())


def discover_dataset_root(candidate: str | Path) -> Path:
    candidate_path = Path(candidate).expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {candidate_path}")

    root_names = {normalize_class_name(path.name) for path in candidate_path.iterdir() if path.is_dir()}
    if set(PAPER_CLASS_ORDER).issubset(root_names):
        return candidate_path

    for subdir_name in ("data", "raw", "dataset", "datasets"):
        subdir = candidate_path / subdir_name
        if not subdir.exists() or not subdir.is_dir():
            continue

        subdir_names = {normalize_class_name(path.name) for path in subdir.iterdir() if path.is_dir()}
        if set(PAPER_CLASS_ORDER).issubset(subdir_names):
            return subdir

    raise FileNotFoundError(
        "Could not find the five berry class folders under "
        f"{candidate_path}. Expected classes: {', '.join(PAPER_CLASS_ORDER)}"
    )


def class_directories(dataset_root: str | Path) -> dict[str, Path]:
    root = discover_dataset_root(dataset_root)
    directories: dict[str, Path] = {}

    for path in root.iterdir():
        if not path.is_dir():
            continue
        canonical_name = normalize_class_name(path.name)
        if canonical_name in PAPER_CLASS_ORDER:
            directories[canonical_name] = path

    missing = [name for name in PAPER_CLASS_ORDER if name not in directories]
    if missing:
        raise FileNotFoundError(
            "Missing class folders: " + ", ".join(missing)
        )

    return directories


def scan_dataset(dataset_root: str | Path) -> tuple[tuple[SampleRecord, ...], tuple[str, ...]]:
    directories = class_directories(dataset_root)
    records: list[SampleRecord] = []

    for label, class_name in enumerate(PAPER_CLASS_ORDER):
        class_dir = directories[class_name]
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for image_path in image_paths:
            records.append(SampleRecord(path=image_path, label=label, class_name=class_name))

    return tuple(records), tuple(PAPER_CLASS_ORDER)


def split_records(
    records: Sequence[SampleRecord],
    seed: int = 42,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> tuple[tuple[SampleRecord, ...], tuple[SampleRecord, ...], tuple[SampleRecord, ...]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    indices = np.arange(len(records))
    labels = np.array([record.label for record in records])

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = labels[train_val_indices]
    val_ratio_within_train_val = val_size / (train_size + val_size)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_within_train_val,
        random_state=seed,
        stratify=train_val_labels,
    )

    def gather(selected_indices: Iterable[int]) -> tuple[SampleRecord, ...]:
        return tuple(records[index] for index in selected_indices)

    return gather(train_indices), gather(val_indices), gather(test_indices)


def compute_class_weights(records: Sequence[SampleRecord]) -> dict[int, float]:
    labels = np.array([record.label for record in records], dtype=np.int32)
    return compute_class_weights_from_labels(labels, num_classes=len(PAPER_CLASS_ORDER))


def compute_class_weights_from_labels(labels: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    return {
        index: float(total / (num_classes * count))
        for index, count in enumerate(counts)
        if count > 0
    }


def build_augmentation_model() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="paper_augmentation",
    )


def _decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: tuple[int, int]) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_tf_dataset(
    records: Sequence[SampleRecord],
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    training: bool = False,
    seed: int = 42,
    augmentation: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    path_strings = [str(record.path) for record in records]
    labels = [record.label for record in records]

    dataset = tf.data.Dataset.from_tensor_slices((path_strings, labels))
    if training:
        dataset = dataset.shuffle(len(path_strings), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda path, label: _decode_and_resize(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training and augmentation is not None:
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_paper_datasets(
    dataset_root: str | Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    seed: int = 42,
) -> DatasetBundle:
    records, class_names = scan_dataset(dataset_root)
    train_records, val_records, test_records = split_records(records, seed=seed)
    class_weights = compute_class_weights(train_records)
    augmentation = build_augmentation_model()

    train_ds = make_tf_dataset(
        train_records,
        image_size=image_size,
        batch_size=batch_size,
        training=True,
        seed=seed,
        augmentation=augmentation,
    )
    val_ds = make_tf_dataset(
        val_records,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )
    test_ds = make_tf_dataset(
        test_records,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        seed=seed,
    )

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        class_names=class_names,
        class_weights=class_weights,
        image_size=image_size,
    )


def sorted_directory_class_names(dataset_root: str | Path) -> tuple[str, ...]:
    root = discover_dataset_root(dataset_root)
    class_names = [
        path.name
        for path in root.iterdir()
        if path.is_dir() and normalize_class_name(path.name) in PAPER_CLASS_ORDER
    ]
    return tuple(sorted(class_names))


def make_paper_generators(
    dataset_root: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    validation_split: float = 0.2,
) -> GeneratorBundle:
    root = discover_dataset_root(dataset_root)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split,
    )

    train_generator = train_datagen.flow_from_directory(
        str(root),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed,
    )
    val_generator = val_datagen.flow_from_directory(
        str(root),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    class_names = tuple(
        class_name
        for class_name, _ in sorted(train_generator.class_indices.items(), key=lambda item: item[1])
    )
    class_weights = compute_class_weights_from_labels(
        np.asarray(train_generator.classes),
        num_classes=len(class_names),
    )

    return GeneratorBundle(
        train_generator=train_generator,
        val_generator=val_generator,
        class_names=class_names,
        class_weights=class_weights,
        image_size=image_size,
        validation_split=validation_split,
        dataset_root=root,
    )
