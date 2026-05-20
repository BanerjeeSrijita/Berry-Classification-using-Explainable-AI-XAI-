# VS Code Guide

This is the simplest path if you want to run everything from VS Code on your local machine before pushing to GitHub.

## 1. Open the folder

Open this folder in VS Code:

```text
/Users/srijitab/Downloads/Pie_berry
```

## 2. Create the virtual environment

Open the terminal in VS Code and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the built-in task runner:

- `Terminal` -> `Run Task...`
- choose `Berry: Create venv`
- then `Berry: Install requirements`

## 3. Select the interpreter

In VS Code:

- `Cmd + Shift + P`
- search `Python: Select Interpreter`
- choose:

```text
.venv/bin/python
```

## 4. Check the environment before training

Run:

```bash
python scripts/validate.py --dataset-root dataset
```

Or use the VS Code task:

- `Berry: Validate`

This confirms:

- Python path
- key package imports
- berry dataset folders
- image counts per class

## 5. Train the paper implementation

Run:

```bash
python scripts/train.py \
  --dataset-root dataset \
  --model proposed \
  --output-dir outputs/paper_proposed
```

Or use the VS Code task:

- `Berry: Train`

## 6. Compare your results with the paper

Run:

```bash
python scripts/evaluate.py \
  --run-dir outputs/paper_proposed
```

Or use the VS Code task:

- `Berry: Evaluate`

## 7. Generate one Grad-CAM sample

Run:

```bash
python scripts/gradcam.py \
  --model-path outputs/paper_proposed/best_model.h5 \
  --image-path dataset/blueberry/0.jpg \
  --output-path outputs/paper_proposed/gradcam_blueberry_0.png
```

Or use the VS Code task:

- `Berry: Grad-CAM`

## 8. Files you should inspect before GitHub upload

Paper implementation outputs:

- `outputs/paper_proposed/metrics.json`
- `outputs/paper_proposed/classification_report.txt`
- `outputs/paper_proposed/confusion_matrix.png`
- `outputs/paper_proposed/training_curves.png`
- `outputs/paper_proposed/validation_predictions.csv`
- `outputs/paper_proposed/gradcam_blueberry_0.png`

Reference notebook outputs:

- `outputs/reference_notebook/berry_features.csv`
- `outputs/reference_notebook/classification_report.txt`
- `outputs/reference_notebook/confusion_matrix.png`

## 9. What to push to GitHub

Push:

- source code
- docs
- notebooks
- `.vscode` tasks if you want them in the repo

Do not push:

- raw dataset folders
- the large proceedings PDF if you want a lighter repo
- generated training outputs unless you intentionally want sample artifacts in the repo

## 10. Minimal Git steps

```bash
git init
git add .
git commit -m "Initial berry classification reproduction repo"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
