#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from berry_xai.data import GeneratorBundle, display_class_name, make_paper_generators
from berry_xai.models import compile_model, get_model, recommended_image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-aligned berry classifier or one of the baseline models."
    )
    parser.add_argument("--dataset-root", default=".", help="Dataset root containing the berry class folders.")
    parser.add_argument(
        "--model",
        default="proposed",
        choices=["proposed", "vgg16", "vgg19", "inceptionv3", "resnet101"],
        help="Model to train.",
    )
    parser.add_argument("--output-dir", default="outputs/paper_proposed", help="Directory for training artifacts.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square image size. Defaults to 224 to match the paper notebook.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation fraction used by flow_from_directory. Default: 0.2",
    )
    parser.add_argument(
        "--backbone-trainable",
        action="store_true",
        help="Allow baseline backbones to fine-tune instead of staying frozen.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "history.csv", index=False)

    epochs = np.arange(1, len(history_frame) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history_frame["accuracy"], "bo-", label="Training Accuracy")
    axes[0].plot(epochs, history_frame["val_accuracy"], "r^-", label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history_frame["loss"], "bo-", label="Training Loss")
    axes[1].plot(epochs, history_frame["val_loss"], "r^-", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_summary(model: tf.keras.Model, output_dir: Path) -> None:
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def collect_predictions(model: tf.keras.Model, generator: object) -> tuple[np.ndarray, np.ndarray]:
    generator.reset()
    probabilities = model.predict(generator, verbose=0)
    return probabilities, np.argmax(probabilities, axis=1)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    labels = [display_class_name(name) for name in class_names]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_predictions_csv(
    bundle: GeneratorBundle,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    filepaths = getattr(bundle.val_generator, "filepaths", None)
    if filepaths is None:
        filepaths = [
            str(bundle.dataset_root / relative_path)
            for relative_path in getattr(bundle.val_generator, "filenames")
        ]

    rows = []
    for image_path, true_label, predicted_label in zip(filepaths, y_true, y_pred):
        rows.append(
            {
                "image_path": str(image_path),
                "true_label_index": int(true_label),
                "true_label_name": bundle.class_names[int(true_label)],
                "predicted_label_index": int(predicted_label),
                "predicted_label_name": bundle.class_names[int(predicted_label)],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_callbacks(output_dir: Path) -> tuple[list[tf.keras.callbacks.Callback], Path]:
    checkpoint_path = output_dir / "best_model.h5"
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks, checkpoint_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)

    image_size_value = args.image_size or recommended_image_size(args.model)
    image_size = (image_size_value, image_size_value)

    bundle = make_paper_generators(
        dataset_root=args.dataset_root,
        image_size=image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        validation_split=args.validation_split,
    )

    model = get_model(
        model_name=args.model,
        input_shape=(image_size_value, image_size_value, 3),
        num_classes=len(bundle.class_names),
        backbone_trainable=args.backbone_trainable,
    )
    model = compile_model(model, learning_rate=args.learning_rate)
    save_model_summary(model, output_dir)

    config = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "model": args.model,
        "image_size": image_size_value,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "validation_split": args.validation_split,
        "backbone_trainable": args.backbone_trainable,
        "class_names": list(bundle.class_names),
        "class_weights": bundle.class_weights,
        "train_count": int(bundle.train_generator.samples),
        "validation_count": int(bundle.val_generator.samples),
        "evaluation_split": "validation",
    }
    write_json(output_dir / "run_config.json", config)

    callbacks, checkpoint_path = build_callbacks(output_dir)
    history = model.fit(
        bundle.train_generator,
        validation_data=bundle.val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    save_history(history, output_dir)

    validation_loss, validation_accuracy = model.evaluate(bundle.val_generator, verbose=0)
    _, y_pred = collect_predictions(model, bundle.val_generator)
    y_true = np.asarray(bundle.val_generator.classes)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=[display_class_name(name) for name in bundle.class_names],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=[display_class_name(name) for name in bundle.class_names],
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    write_json(output_dir / "classification_report.json", report_dict)
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(bundle.class_names),
        output_path=output_dir / "confusion_matrix.png",
    )
    save_predictions_csv(bundle, y_true, y_pred, output_dir / "validation_predictions.csv")

    metrics = {
        "evaluation_split": "validation",
        "validation_loss": float(validation_loss),
        "validation_accuracy": float(validation_accuracy),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1_score": float(report_dict["macro avg"]["f1-score"]),
    }
    write_json(output_dir / "metrics.json", metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
