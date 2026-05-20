# Explainable Deep Learning in Berry Classification

This repository turns the local workspace into a GitHub-ready project around:

1. the original `Pie_XAI.ipynb` notebook already present in this folder
2. the paper notebook `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb`
3. a scriptified version of **"Explainable Deep Learning in Berry Classification Through Attention Mechanisms and Grad-CAM"**

The important caveat is that these are **not the same workflow**.

- `Pie_XAI.ipynb` is a notebook pipeline that builds an untrained lightweight CNN, uses it as a feature extractor, applies PCA, trains a Random Forest, and then runs SHAP and LIME.
- `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb` is the notebook now being treated as the exact paper implementation.
- `scripts/train.py` turns that paper notebook into a CLI-friendly training script for local, VS Code, or GPU server runs.

## What is included

- `Pie_XAI.ipynb`: the original notebook artifact from the workspace
- `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb`: the paper notebook used as the source of truth for the main CNN path
- `scripts/run_notebook.py`: a cleaned, local, runnable adaptation of the notebook workflow
- `scripts/train.py`: scriptified version of the exact paper notebook workflow
- `scripts/gradcam.py`: Grad-CAM generation for trained paper models
- `scripts/validate.py`: quick local readiness check for VS Code or terminal use
- `scripts/evaluate.py`: direct metric comparison against the paper's reported values
- `src/berry_xai/`: reusable data, model, generator, and Grad-CAM helpers
- `docs/implementation_notes.md`: audit trail of what the paper says, what each notebook does, and what was scriptified
- `docs/vscode_guide.md`: step-by-step VS Code workflow for local execution and GitHub prep

## Important reproducibility notes

1. There are two different notebook lineages in this repo.
   - The notebook never calls `model.fit(...)`, so its CNN feature extractor is untrained.
   - The paper notebook does call `model.fit(...)` and trains a multi-branch attention CNN with Grad-CAM.

2. The main paper script now mirrors `Copy_of_PIE_berry_Multi_branch_XAI_.ipynb`.
   - image size: `224x224`
   - `ImageDataGenerator(..., validation_split=0.2)`
   - two attention branches: `3x3` and `5x5`
   - dropout `0.3` in the branches and `0.4` after merge
   - `last_conv` as the Grad-CAM layer
   - Adam with learning rate `0.0003`

3. The paper text still has a split inconsistency.
   - The chapter text mentions both `2016/503` and a train-validation split setup.
   - The exact notebook path in this repo uses an **80/20 generator split**, because that is what the notebook code actually does.

## Expected dataset layout

The local workspace already contains the dataset at the project root:

```text
dataset/
├── Tayberry/
├── Wineberry/
├── black berry/
├── blueberry/
└── white mulberry/
```

The helper code also accepts any dataset root that contains those class folders.

## Project layout

```text
.
├── Copy_of_PIE_berry_Multi_branch_XAI_.ipynb
├── Pie_XAI.ipynb
├── README.md
├── environment.yml
├── dataset/
├── docs/
│   ├── implementation_notes.md
│   └── vscode_guide.md
├── requirements.txt
├── scripts/
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── run_notebook.py
│   ├── train.py
│   └── validate.py
├── src/
│   └── berry_xai/
│       ├── __init__.py
│       ├── data.py
│       ├── gradcam.py
│       └── models.py
└── outputs/
```

## Setup

With `venv` and `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With `conda`:

```bash
conda env create -f environment.yml
conda activate berry-xai
```

On Linux GPU systems, the environment file now installs TensorFlow with the official `and-cuda` extra so the required CUDA user-space libraries are pulled into the environment.

## Recommended run order

1. Run `python scripts/validate.py --dataset-root dataset`
2. Run `python scripts/train.py --dataset-root dataset --model proposed --output-dir outputs/paper_proposed`
3. Run `python scripts/evaluate.py --run-dir outputs/paper_proposed`
4. Run `python scripts/gradcam.py --model-path outputs/paper_proposed/best_model.h5 --image-path dataset/Tayberry/1.jpg --output-path outputs/paper_proposed/gradcam_tayberry_1.png`

## Quick environment check

Before training, verify that VS Code or your terminal is seeing the dataset and packages correctly:

```bash
python scripts/validate.py --dataset-root dataset
```

The validator reports:

- installed packages
- dataset counts
- TensorFlow GPU visibility

If you prefer, follow the local workflow in [vscode_guide.md](/Users/srijitab/Downloads/Pie_berry/docs/vscode_guide.md).

## Local quick start

If you are running on your own machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/validate.py --dataset-root dataset

python scripts/train.py \
  --dataset-root dataset \
  --model proposed \
  --output-dir outputs/paper_proposed

python scripts/evaluate.py \
  --run-dir outputs/paper_proposed

python scripts/gradcam.py \
  --model-path outputs/paper_proposed/best_model.h5 \
  --image-path dataset/Tayberry/1.jpg \
  --output-path outputs/paper_proposed/gradcam_tayberry_1.png
```

## GPU server quick start

If you are running on a shared Linux GPU machine or GPU server:

```bash
cd /path/to/Pie_berry

conda deactivate 2>/dev/null || true
conda env create -n berry-xai-gpu -f environment.yml
conda activate berry-xai-gpu

python -m pip install -r requirements.txt

python scripts/validate.py --dataset-root dataset
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

CUDA_VISIBLE_DEVICES=1 python scripts/train.py \
  --dataset-root dataset \
  --model proposed \
  --output-dir outputs/paper_proposed

python scripts/evaluate.py \
  --run-dir outputs/paper_proposed

python scripts/gradcam.py \
  --model-path outputs/paper_proposed/best_model.h5 \
  --image-path dataset/Tayberry/1.jpg \
  --output-path outputs/paper_proposed/gradcam_tayberry_1.png
```

Notes for shared systems:

- If `conda env remove` fails with `.nfs...` warnings, do not fight the old environment. Create a new one with a different name such as `berry-xai-gpu`.
- If TensorFlow prints `GPUs: []`, the environment is still CPU-only.
- If you accidentally suspend training and see `[1]+ Stopped`, run `fg` to resume it or `kill %1` to stop it.
- The messages `Found 2016 images belonging to 5 classes.` and `Found 503 images belonging to 5 classes.` are expected. They come from the notebook's `validation_split=0.2`.

## If GPU is not detected on Linux

The repository environment uses TensorFlow's `and-cuda` extra on Linux. If TensorFlow still does not see a GPU, try:

```bash
pushd $(dirname $(python -c 'print(__import__("tensorflow").__file__)'))
ln -svf ../nvidia/*/lib/*.so* .
popd

ln -sf $(find $(dirname $(dirname $(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"))/*/bin/) -name ptxas -print -quit) $CONDA_PREFIX/bin/ptxas

python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Verify the results

After training, inspect:

- `outputs/paper_proposed/metrics.json`
- `outputs/paper_proposed/classification_report.txt`
- `outputs/paper_proposed/confusion_matrix.png`
- `outputs/paper_proposed/training_curves.png`
- `outputs/paper_proposed/validation_predictions.csv`
- `outputs/paper_proposed/gradcam_tayberry_1.png`

Useful commands:

```bash
cat outputs/paper_proposed/metrics.json
cat outputs/paper_proposed/classification_report.txt
ls -lh outputs/paper_proposed
```

The paper-aligned path evaluates on the validation subset created by `ImageDataGenerator(..., validation_split=0.2)`.

## Expected paper-style behavior

When the main paper workflow is running correctly, you should see:

- `224x224` images
- training and validation generators from `flow_from_directory(...)`
- `2016` training images and `503` validation images
- a model with two branches: `3x3` and `5x5`
- `last_conv` used for Grad-CAM
- `best_model.h5` saved during training

## Run the exact notebook

If you want to use the notebook directly instead of the scriptified path, open:

- `PIE_berry_Multi_branch_XAI_.ipynb`

The main script `scripts/train.py` is intended to mirror that notebook for repeatable CLI runs.

## Run the notebook workflow

This reproduces the notebook logic as a script, with only the minimum fixes needed to run locally.

```bash
python scripts/run_notebook.py \
  --dataset-root dataset \
  --output-dir outputs/reference_notebook
```

Outputs include:

- `berry_features.csv`
- model summary text
- Random Forest classification report
- confusion matrix
- PCA visualizations
- SHAP summary plot if `shap` is installed
- LIME explanation plot if `lime` is installed

## Train the paper implementation

```bash
python scripts/train.py \
  --dataset-root dataset \
  --model proposed \
  --output-dir outputs/paper_proposed
```

Default paper script settings:

- image size: `224x224`
- batch size: `32`
- epochs: `50`
- runs the full `50` epochs by default
- optimizer: Adam
- learning rate: `0.0003`
- validation split: `0.2`
- callbacks:
  - `ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-6)`
  - `ModelCheckpoint(save_best_only=True)`

## Train baseline models

The same training script can run the paper baselines:

```bash
python scripts/train.py --dataset-root dataset --model vgg16 --output-dir outputs/vgg16
python scripts/train.py --dataset-root dataset --model vgg19 --output-dir outputs/vgg19
python scripts/train.py --dataset-root dataset --model inceptionv3 --output-dir outputs/inceptionv3
python scripts/train.py --dataset-root dataset --model resnet101 --output-dir outputs/resnet101
```

## Generate Grad-CAM

```bash
python scripts/gradcam.py \
  --model-path outputs/paper_proposed/best_model.h5 \
  --image-path "dataset/blueberry/0.jpg" \
  --output-path outputs/paper_proposed/gradcam_blueberry_0.png
```

## Compare your run with the paper

```bash
python scripts/evaluate.py \
  --run-dir outputs/paper_proposed
```

This checks your exported `metrics.json` and `classification_report.json` against the paper's reported values and marks each metric as `close` or `far`.
The script reports whether the run was evaluated on a validation or test split.

## Prepare for GitHub

Before pushing, make sure:

- the script names are the final uniform set:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/validate.py`
  - `scripts/gradcam.py`
  - `scripts/run_notebook.py`
- the README matches the commands you actually ran
- the dataset is not committed
- large generated outputs are not committed unless you intentionally want sample artifacts in the repo

Typical Git commands:

```bash
git init
git add .
git commit -m "Initial berry classification reproduction repo"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Sources

- Paper DOI: <https://doi.org/10.1007/978-3-032-07336-5_1>
- Dataset referenced in the paper: <https://www.kaggle.com/datasets/aelchimminut/fruits262>
