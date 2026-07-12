"""Shared Keras ResNet-50 implementation used by matrix notebooks and the CLI runner."""
from __future__ import annotations

from pathlib import Path


def configure_tensorflow() -> None:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def build_resnet50_model(input_size=(224, 224), pretrained=True):
    import tensorflow as tf
    configure_tensorflow()
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=tuple(input_size) + (3,),
    )
    inputs = tf.keras.Input(shape=tuple(input_size) + (3,))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs, name="matrix_resnet50"), backbone


def set_head_training(backbone) -> None:
    backbone.trainable = False


def set_fine_tuning(backbone, start_layer="conv4_block5_1_conv") -> None:
    backbone.trainable = True
    enabled = False
    for layer in backbone.layers:
        if layer.name == start_layer:
            enabled = True
        layer.trainable = enabled and layer.__class__.__name__ != "BatchNormalization"


def make_dataset(rows, input_size=(224, 224), batch_size=16, shuffle=False, seed=42):
    import tensorflow as tf
    paths = [row["processed_path"] for row in rows]
    labels = [float(row["label"]) for row in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(tf.cast(image, tf.float32), input_size)
        image.set_shape((*input_size, 3))
        return image, label

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def predict_validation(model, loader):
    import numpy as np
    labels = []
    for _, batch_labels in loader:
        labels.extend(np.asarray(batch_labels).reshape(-1).astype(int).tolist())
    probabilities = np.asarray(model.predict(loader, verbose=0)).reshape(-1).astype(float).tolist()
    return labels, probabilities


def load_keras_checkpoint(path: Path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)
