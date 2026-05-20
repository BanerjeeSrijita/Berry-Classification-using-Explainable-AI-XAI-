"""Helpers for berry-classification reproduction workflows."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DatasetBundle",
    "GeneratorBundle",
    "PAPER_CLASS_ORDER",
    "build_baseline_model",
    "build_proposed_model",
    "compile_model",
    "discover_dataset_root",
    "make_paper_generators",
]

def __getattr__(name: str):
    if name in {
        "DatasetBundle",
        "GeneratorBundle",
        "PAPER_CLASS_ORDER",
        "discover_dataset_root",
        "make_paper_generators",
    }:
        module = import_module(".data", __name__)
        return getattr(module, name)

    if name in {
        "build_baseline_model",
        "build_proposed_model",
        "compile_model",
    }:
        module = import_module(".models", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
