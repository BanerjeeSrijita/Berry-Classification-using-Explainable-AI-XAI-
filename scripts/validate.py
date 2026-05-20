#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REQUIRED_PACKAGES = [
    "tensorflow",
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
    "cv2",
    "PIL",
    "tqdm",
]

OPTIONAL_PACKAGES = [
    "shap",
    "lime",
    "visualkeras",
]

EXPECTED_CLASS_NAMES = {
    "tayberry",
    "wineberry",
    "black berry",
    "blueberry",
    "white mulberry",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local environment and dataset readiness.")
    parser.add_argument("--dataset-root", default=".", help="Dataset root or project root.")
    return parser.parse_args()


def package_status(package_name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return {"installed": True, "version": str(version)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"installed": False, "version": f"missing ({exc.__class__.__name__})"}


def tensorflow_gpu_summary() -> dict[str, object]:
    try:
        tensorflow = importlib.import_module("tensorflow")
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "tensorflow_imported": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "visible_gpus": [],
        }

    try:
        devices = tensorflow.config.list_physical_devices("GPU")
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "tensorflow_imported": True,
            "error": f"{exc.__class__.__name__}: {exc}",
            "visible_gpus": [],
        }

    return {
        "tensorflow_imported": True,
        "visible_gpus": [device.name for device in devices],
        "gpu_count": len(devices),
    }


def discover_dataset_root(candidate: str | Path) -> Path:
    candidate_path = Path(candidate).expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {candidate_path}")

    child_dir_names = {path.name.lower() for path in candidate_path.iterdir() if path.is_dir()}
    if EXPECTED_CLASS_NAMES.issubset(child_dir_names):
        return candidate_path

    for subdir_name in ("data", "raw", "dataset", "datasets"):
        subdir = candidate_path / subdir_name
        if not subdir.exists() or not subdir.is_dir():
            continue
        subdir_names = {path.name.lower() for path in subdir.iterdir() if path.is_dir()}
        if EXPECTED_CLASS_NAMES.issubset(subdir_names):
            return subdir

    raise FileNotFoundError(
        "Could not find the berry class folders under "
        f"{candidate_path}. Expected: {', '.join(sorted(EXPECTED_CLASS_NAMES))}"
    )


def class_directories(dataset_root: str | Path) -> dict[str, Path]:
    root = discover_dataset_root(dataset_root)
    directories: dict[str, Path] = {}
    for path in root.iterdir():
        if path.is_dir() and path.name.lower() in EXPECTED_CLASS_NAMES:
            directories[path.name.lower()] = path
    return directories


def dataset_summary(dataset_root: Path) -> dict[str, int]:
    directories = class_directories(dataset_root)
    summary: dict[str, int] = {}
    for class_name, class_dir in directories.items():
        count = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        summary[class_name] = count
    return summary


def main() -> None:
    args = parse_args()
    dataset_root = discover_dataset_root(args.dataset_root)

    required = {name: package_status(name) for name in REQUIRED_PACKAGES}
    optional = {name: package_status(name) for name in OPTIONAL_PACKAGES}
    summary = dataset_summary(dataset_root)

    payload = {
        "python_executable": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(dataset_root),
        "required_packages": required,
        "optional_packages": optional,
        "tensorflow_gpu": tensorflow_gpu_summary(),
        "dataset_counts": summary,
        "dataset_total": sum(summary.values()),
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
