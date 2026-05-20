# Implementation Notes

This document keeps the repository honest about what comes from each notebook artifact and what had to be inferred before the paper notebook was found.

## Side-by-side summary

| Aspect | Paper / Paper Notebook | `Pie_XAI.ipynb` | Repo handling |
| --- | --- | --- | --- |
| Core model | Lightweight multi-branch CNN with attention | Simple 3-layer CNN | `train.py` now mirrors `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb` |
| Training style | End-to-end supervised training | No `fit(...)` call at all | Both preserved: paper training path and original notebook path |
| Explanation method | Grad-CAM | SHAP and LIME on Random Forest after PCA | Both supported: Grad-CAM for paper model, SHAP/LIME for notebook workflow |
| Data split | Notebook code uses generator `validation_split=0.2` | 80/20 split after PCA | Paper script now uses the notebook's 80/20 generator split |
| Baselines | VGG16, VGG19, InceptionV3, ResNet101 | Not present | Added as configurable baselines in the paper training script |
| Feature extraction | Learned by trained attention CNN | Features taken from an untrained CNN dense layer | Notebook adaptation keeps the original behavior and documents it clearly |

## Notebook-specific issues corrected in the cleaned script

The script `scripts/run_notebook.py` makes only the minimum practical fixes needed to run the notebook logic locally:

1. Replaced the hard-coded Colab path `/content/drive/MyDrive/Pie_berry` with a CLI dataset path.
2. Fixed the `print("Features saved locally as 'berry_features.csv")` quote typo.
3. Replaced the undefined `X_test_df` reference with the actual PCA dataframe.
4. Removed notebook-only `!pip install ...` magics.
5. Normalized class-name handling so that labels remain consistent across PCA, Random Forest, SHAP, and LIME outputs.

## What changed after finding `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb`

The repo originally used an inferred implementation from the chapter text. After finding the paper notebook, the main paper script was aligned to the notebook code:

- `IMAGE_SIZE = (224, 224)`
- `BATCH_SIZE = 32`
- `validation_split = 0.2`
- `ImageDataGenerator` augmentation:
  - `rescale=1./255`
  - `shear_range=0.2`
  - `zoom_range=0.2`
  - `rotation_range=30`
  - `width_shift_range=0.1`
  - `height_shift_range=0.1`
  - `horizontal_flip=True`
- Proposed model:
  - branch 1: `Conv2D(32, 3x3)` -> `BatchNormalization` -> `Dropout(0.3)` -> attention
  - branch 2: `Conv2D(32, 5x5)` -> `BatchNormalization` -> `Dropout(0.3)` -> attention
  - concatenate branches
  - `Conv2D(64, 3x3, name="last_conv")` -> `BatchNormalization` -> `Dropout(0.4)`
  - `GlobalAveragePooling2D`
  - `Dense(num_classes, softmax)`
- Optimizer:
  - Adam with `learning_rate=0.0003`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-07`
- Callbacks:
  - `ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6, verbose=1)`
  - `ModelCheckpoint(..., save_best_only=True, monitor="val_accuracy")`

The current CLI script intentionally does not use early stopping so it runs the full `50` epochs and produces the full training curve.

## Remaining caveat

The notebook evaluates on the validation subset created by `flow_from_directory(..., validation_split=0.2)`, not on a separately materialized held-out test folder. The evaluation script in this repo reports which split was used so the repo stays explicit about that.

## Why both pipelines are worth keeping

- The notebook is still useful because it captures the exact local reference artifact the project started with.
- The paper notebook is useful because it captures the end-to-end CNN code path the chapter is based on.

Keeping both makes the repo transparent and much easier to maintain, cite, and extend.
