#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, layers, models
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from berry_xai.data import discover_dataset_root, normalize_class_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleaned local adaptation of the original Pie_XAI.ipynb workflow."
    )
    parser.add_argument("--dataset-root", default=".", help="Root containing the five berry class folders.")
    parser.add_argument("--output-dir", default="outputs/reference_notebook", help="Directory for notebook artifacts.")
    parser.add_argument("--image-size", type=int, default=128, help="Square input size used in the notebook.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used in the notebook.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Random Forest holdout fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lime-index", type=int, default=0, help="Test-sample index to explain with LIME.")
    return parser.parse_args()


def create_custom_cnn_functional(input_shape: tuple[int, int, int] = (128, 128, 3), num_classes: int = 5) -> models.Model:
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)


def pretty_notebook_name(name: str) -> str:
    canonical = normalize_class_name(name)
    if canonical == "blackberry":
        return "blackberry"
    return canonical


def save_model_summary(model: models.Model, output_dir: Path) -> None:
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (output_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def save_visualkeras_diagram(model: models.Model, output_dir: Path) -> None:
    try:
        import visualkeras
        from PIL import ImageFont
    except ImportError:
        return

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = None

    visualkeras.layered_view(
        model,
        to_file=str(output_dir / "cnn_visual_block.png"),
        legend=True,
        draw_volume=True,
        spacing=100,
        font=font,
    )


def sorted_class_directories(dataset_root: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in dataset_root.iterdir()
        if entry.is_dir() and normalize_class_name(entry.name) in {
            "blackberry",
            "blueberry",
            "tayberry",
            "white mulberry",
            "wineberry",
        }
    )


def extract_features(
    feature_extractor: models.Model,
    dataset_root: Path,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    class_names = sorted_class_directories(dataset_root)
    features: list[np.ndarray] = []
    labels: list[int] = []
    image_paths: list[str] = []

    for label_index, class_name in enumerate(class_names):
        class_folder = dataset_root / class_name
        for img_file in tqdm(sorted(os.listdir(class_folder)), desc=f"Processing {class_name}"):
            img_path = class_folder / img_file
            if not img_path.is_file():
                continue
            try:
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                feature = feature_extractor.predict(img_array, verbose=0)
                features.append(feature.flatten())
                labels.append(label_index)
                image_paths.append(str(img_path))
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Skipping {img_path}: {exc}")

    return np.asarray(features), np.asarray(labels), class_names, image_paths


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    labels = [pretty_notebook_name(name) for name in class_names]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_distribution_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    frame.groupby("class_names").size().plot(kind="barh", color=sns.color_palette("Dark2"))
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_violin_plot(frame: pd.DataFrame, output_path: Path) -> None:
    figsize = (12, 1.2 * len(frame["class_names"].unique()))
    plt.figure(figsize=figsize)
    sns.violinplot(frame, x="PC1", y="class_names", inner="box", palette="Dark2")
    sns.despine(top=True, right=True, bottom=True, left=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def maybe_run_shap(
    rf_model: RandomForestClassifier,
    x_test_pca: np.ndarray,
    x_test_pca_only: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import shap
    except ImportError:
        return

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(x_test_pca)

    plt.figure()
    shap.summary_plot(shap_values, x_test_pca_only, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def maybe_run_lime(
    rf_model: RandomForestClassifier,
    x_train_pca: np.ndarray,
    x_test_pca: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    lime_index: int,
    output_dir: Path,
) -> None:
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        return

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=x_train_pca,
        feature_names=[f"PC{i + 1}" for i in range(x_train_pca.shape[1])],
        class_names=[pretty_notebook_name(name) for name in class_names],
        mode="classification",
        verbose=True,
    )

    safe_index = max(0, min(lime_index, len(x_test_pca) - 1))
    sample = x_test_pca[safe_index]

    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=rf_model.predict_proba,
        num_features=10,
    )

    figure = explanation.as_pyplot_figure()
    figure.suptitle(
        "LIME Explanation for Test Sample "
        f"{safe_index} (True label: {pretty_notebook_name(class_names[y_test[safe_index]])})"
    )
    figure.tight_layout()
    figure.savefig(output_dir / "lime_explanation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)
    dataset_root = discover_dataset_root(args.dataset_root)

    initial_class_names = sorted_class_directories(dataset_root)
    (output_dir / "dataset_class_names.json").write_text(
        json.dumps(initial_class_names, indent=2),
        encoding="utf-8",
    )

    model = create_custom_cnn_functional(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=len(initial_class_names),
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    feature_extractor = models.Model(inputs=model.input, outputs=model.layers[-2].output)

    save_model_summary(model, output_dir)
    save_visualkeras_diagram(model, output_dir)

    features, labels, class_names, image_paths = extract_features(
        feature_extractor=feature_extractor,
        dataset_root=dataset_root,
        image_size=(args.image_size, args.image_size),
    )

    features_frame = pd.DataFrame(features)
    features_frame["label"] = labels
    features_frame["image_path"] = image_paths
    features_csv_path = output_dir / "berry_features.csv"
    features_frame.to_csv(features_csv_path, index=False)

    x = features_frame.drop(columns=["label", "image_path"]).values
    y = features_frame["label"].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=0.95)
    x_pca = pca.fit_transform(x_scaled)

    x_train_pca, x_test_pca, y_train, y_test = train_test_split(
        x_pca,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    rf_model = RandomForestClassifier(random_state=args.seed)
    rf_model.fit(x_train_pca, y_train)
    y_pred = rf_model.predict(x_test_pca)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=[pretty_notebook_name(name) for name in class_names],
        zero_division=0,
    )
    report_json = classification_report(
        y_test,
        y_pred,
        target_names=[pretty_notebook_name(name) for name in class_names],
        zero_division=0,
        output_dict=True,
    )

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "classification_report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    x_test_pca_df = pd.DataFrame(
        x_test_pca,
        columns=[f"PC{i + 1}" for i in range(x_test_pca.shape[1])],
    )
    x_test_pca_df["label"] = y_test
    x_test_pca_df["class_names"] = x_test_pca_df["label"].apply(lambda value: pretty_notebook_name(class_names[value]))

    save_distribution_plot(x_test_pca_df, output_dir / "class_distribution.png")
    save_violin_plot(x_test_pca_df, output_dir / "pc1_violin.png")

    x_test_pca_only = x_test_pca_df.drop(columns=["label", "class_names"])
    maybe_run_shap(rf_model, x_test_pca, x_test_pca_only, output_dir)
    maybe_run_lime(
        rf_model=rf_model,
        x_train_pca=x_train_pca,
        x_test_pca=x_test_pca,
        y_test=y_test,
        class_names=class_names,
        lime_index=args.lime_index,
        output_dir=output_dir,
    )

    with open(output_dir / "reference_pipeline.pkl", "wb") as handle:
        pickle.dump(
            {
                "scaler": scaler,
                "pca": pca,
                "random_forest": rf_model,
                "class_names": class_names,
            },
            handle,
        )

    run_summary = {
        "dataset_root": str(dataset_root),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "feature_shape": list(features.shape),
        "pca_shape": list(x_pca.shape),
        "note": (
            "This script preserves the notebook's original modeling idea, including the "
            "fact that the CNN feature extractor is never trained before feature export."
        ),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
