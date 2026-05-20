from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def channel_attention_block(
    inputs: tf.Tensor,
    reduction: int = 8,
    name: str = "attention",
) -> tf.Tensor:
    channels = inputs.shape[-1]
    reduced_channels = max(1, int(channels) // reduction)

    x = layers.GlobalAveragePooling2D(name=f"{name}_gap")(inputs)
    x = layers.Dense(reduced_channels, activation="relu", name=f"{name}_fc1")(x)
    x = layers.Dense(int(channels), activation="sigmoid", name=f"{name}_fc2")(x)
    x = layers.Reshape((1, 1, int(channels)), name=f"{name}_reshape")(x)
    return layers.Multiply(name=f"{name}_scale")([inputs, x])


def attention_branch(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int,
    branch_name: str,
    attention_reduction: int = 8,
    dropout_rate: float = 0.3,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="same",
        activation="relu",
        name=f"{branch_name}_conv",
    )(inputs)
    x = layers.BatchNormalization(name=f"{branch_name}_bn")(x)
    x = layers.Dropout(dropout_rate, name=f"{branch_name}_dropout")(x)
    x = channel_attention_block(
        x,
        reduction=attention_reduction,
        name=f"{branch_name}_attention",
    )
    return x


def build_proposed_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 5,
    branch_filters: int = 32,
    shared_filters: int = 64,
    branch_dropout: float = 0.3,
    shared_dropout: float = 0.4,
    attention_reduction: int = 8,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="input_image")

    branch_3x3 = attention_branch(
        inputs,
        filters=branch_filters,
        kernel_size=3,
        branch_name="branch_3x3",
        dropout_rate=branch_dropout,
        attention_reduction=attention_reduction,
    )
    branch_5x5 = attention_branch(
        inputs,
        filters=branch_filters,
        kernel_size=5,
        branch_name="branch_5x5",
        dropout_rate=branch_dropout,
        attention_reduction=attention_reduction,
    )

    x = layers.Concatenate(name="branch_concat")([branch_3x3, branch_5x5])
    x = layers.Conv2D(
        shared_filters,
        3,
        padding="same",
        activation="relu",
        name="last_conv",
    )(x)
    x = layers.BatchNormalization(name="shared_bn")(x)
    x = layers.Dropout(shared_dropout, name="shared_dropout")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="paper_multi_branch_attention_cnn",
    )


def baseline_registry() -> dict[str, tuple[callable, callable, int]]:
    return {
        "vgg16": (
            keras.applications.VGG16,
            keras.applications.vgg16.preprocess_input,
            224,
        ),
        "vgg19": (
            keras.applications.VGG19,
            keras.applications.vgg19.preprocess_input,
            224,
        ),
        "inceptionv3": (
            keras.applications.InceptionV3,
            keras.applications.inception_v3.preprocess_input,
            224,
        ),
        "resnet101": (
            keras.applications.ResNet101,
            keras.applications.resnet.preprocess_input,
            224,
        ),
    }


def recommended_image_size(model_name: str) -> int:
    return 224


def build_baseline_model(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.3,
    backbone_trainable: bool = False,
) -> keras.Model:
    registry = baseline_registry()
    if model_name not in registry:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    builder, preprocessor, _ = registry[model_name]
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Lambda(lambda image: image * 255.0, name="scale_to_imagenet_range")(inputs)
    x = layers.Lambda(preprocessor, name="imagenet_preprocess")(x)
    backbone = builder(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    backbone.trainable = backbone_trainable

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(backbone.output)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=f"{model_name}_baseline")


def get_model(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    backbone_trainable: bool = False,
) -> keras.Model:
    if model_name == "proposed":
        return build_proposed_model(input_shape=input_shape, num_classes=num_classes)

    return build_baseline_model(
        model_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
        backbone_trainable=backbone_trainable,
    )


def compile_model(model: keras.Model, learning_rate: float = 3e-4) -> keras.Model:
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
