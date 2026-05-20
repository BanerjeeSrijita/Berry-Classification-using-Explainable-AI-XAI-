#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_OVERALL = {
    "accuracy": 0.904,
    "macro_precision": 0.902,
    "macro_recall": 0.900,
    "macro_f1_score": 0.900,
}

PAPER_CLASSWISE = {
    "Tayberry": {"precision": 0.85, "recall": 0.94, "f1-score": 0.90},
    "Wineberry": {"precision": 0.93, "recall": 0.76, "f1-score": 0.84},
    "Blackberry": {"precision": 0.88, "recall": 0.98, "f1-score": 0.93},
    "Blueberry": {"precision": 0.97, "recall": 0.91, "f1-score": 0.94},
    "White Mulberry": {"precision": 0.89, "recall": 0.92, "f1-score": 0.91},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trained run against the paper's reported berry-classification metrics."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing metrics.json and classification_report.json from train.py",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.03,
        help="Absolute tolerance for considering a metric 'close' to the paper. Default: 0.03",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status(delta: float, tolerance: float) -> str:
    return "close" if abs(delta) <= tolerance else "far"


def actual_overall_metric(metrics: dict, metric_name: str) -> tuple[str | None, float | None]:
    if metric_name == "accuracy":
        for candidate in ("validation_accuracy", "test_accuracy"):
            if candidate in metrics:
                return candidate, metrics[candidate]
        return None, None
    return metric_name, metrics.get(metric_name)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    metrics = load_json(run_dir / "metrics.json")
    report = load_json(run_dir / "classification_report.json")
    evaluation_split = metrics.get("evaluation_split", "unknown")

    print("Overall metrics")
    print("-" * 60)
    print(f"Run evaluation split: {evaluation_split}")
    print()
    for metric_name, expected in PAPER_OVERALL.items():
        actual_key, actual = actual_overall_metric(metrics, metric_name)
        delta = None if actual is None else actual - expected
        if actual is None:
            print(f"{metric_name:20s} missing")
            continue
        print(
            f"{metric_name:20s} paper={expected:.3f}  run={actual:.3f}  "
            f"delta={delta:+.3f}  {status(delta, args.tolerance)}"
            f"  ({actual_key})"
        )

    print()
    print("Class-wise metrics")
    print("-" * 60)
    for class_name, expected_metrics in PAPER_CLASSWISE.items():
        actual_metrics = report.get(class_name)
        if actual_metrics is None:
            print(f"{class_name}: missing")
            continue

        print(class_name)
        for metric_name, expected in expected_metrics.items():
            actual = actual_metrics.get(metric_name)
            delta = None if actual is None else actual - expected
            if actual is None:
                print(f"  {metric_name:10s} missing")
                continue
            print(
                f"  {metric_name:10s} paper={expected:.3f}  run={actual:.3f}  "
                f"delta={delta:+.3f}  {status(delta, args.tolerance)}"
            )

    print()
    print("Paper reference")
    print("-" * 60)
    print("Overall: accuracy=0.904 precision=0.902 recall=0.900 f1=0.900")
    print("Classes: Tayberry, Wineberry, Blackberry, Blueberry, White Mulberry")


if __name__ == "__main__":
    main()
