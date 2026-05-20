#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from berry_xai.gradcam import save_gradcam_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay for a trained model.")
    parser.add_argument("--model-path", required=True, help="Path to a trained Keras model.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    parser.add_argument("--output-path", required=True, help="Where to save the Grad-CAM overlay.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size used by the model.")
    parser.add_argument("--class-index", type=int, default=None, help="Optional target class index.")
    parser.add_argument("--last-conv-layer", default=None, help="Optional explicit final convolution layer name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    metadata = save_gradcam_overlay(
        image_path=args.image_path,
        model=model,
        output_path=args.output_path,
        image_size=(args.image_size, args.image_size),
        last_conv_layer_name=args.last_conv_layer,
        class_index=args.class_index,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
