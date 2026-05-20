from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def load_image_array(image_path: str | Path, image_size: tuple[int, int]) -> tuple[np.ndarray, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    resized = image.resize(image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    return array, image


def infer_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not find a Conv2D layer for Grad-CAM.")


def make_gradcam_heatmap(
    image_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
) -> tuple[np.ndarray, int]:
    if last_conv_layer_name is None:
        last_conv_layer_name = infer_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (
        tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )
    return heatmap.numpy(), class_index


def overlay_gradcam(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, original_image.size)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image_rgb = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    blended = cv2.addWeighted(image_rgb, 0.6, heatmap, alpha, 0)
    return Image.fromarray(blended)


def save_gradcam_overlay(
    image_path: str | Path,
    model: tf.keras.Model,
    output_path: str | Path,
    image_size: tuple[int, int],
    last_conv_layer_name: str | None = None,
    class_index: int | None = None,
    alpha: float = 0.4,
) -> dict[str, int | str]:
    image_array, original_image = load_image_array(image_path, image_size=image_size)
    heatmap, predicted_index = make_gradcam_heatmap(
        image_array=image_array,
        model=model,
        last_conv_layer_name=last_conv_layer_name,
        class_index=class_index,
    )
    overlay = overlay_gradcam(original_image=original_image, heatmap=heatmap, alpha=alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return {
        "image_path": str(image_path),
        "output_path": str(output_path),
        "predicted_class_index": predicted_index,
        "last_conv_layer_name": last_conv_layer_name or infer_last_conv_layer_name(model),
    }
