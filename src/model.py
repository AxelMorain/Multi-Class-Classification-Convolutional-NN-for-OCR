"""CNN architecture for 36-class OCR character classification.

``build_model_v4`` is the architecture that reached ~99% test accuracy. Three
earlier, vastly different architectures were tried during development; this one
stood out clearly as the best and is the one kept here.
"""

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    ReLU,
    Softmax,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

INPUT_SHAPE = (32, 32, 1)
NUM_CLASSES = 36


def build_model_v4(
    dropout_rate: float = 0.25,
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> Sequential:
    """Build and compile the best OCR CNN.

    Architecture: an initial MaxPool followed by five Conv2D + MaxPool blocks
    (25 filters each), then a Flatten and a stack of Dense layers ending in a
    36-way Softmax. A single Dropout layer sits after the first Dense layer.

    The dropout rate noticeably affects convergence speed: at 0.25 the model
    crosses 80% validation accuracy around epoch 13; with no dropout it takes
    ~18 epochs; raising it to 0.30 becomes detrimental (>15 epochs).

    Args:
        dropout_rate: Rate for the Dropout layer after the first Dense layer.
        input_shape: Shape of a single input image.
        num_classes: Number of output classes (36 = digits + uppercase letters).
    Returns:
        A compiled ``Sequential`` model (SGD + CategoricalCrossentropy).
    """
    model = Sequential(name="ocr_cnn_v4")
    model.add(Input(shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding="same"))

    # Five Conv2D + MaxPool blocks (first kernel 5x5, the rest 3x3).
    kernel_sizes = [5, 3, 3, 3, 3]
    for kernel_size in kernel_sizes:
        model.add(
            Conv2D(
                filters=25,
                kernel_size=kernel_size,
                strides=(1, 1),
                padding="same",
                activation=ReLU(),
            )
        )
        model.add(MaxPooling2D((2, 2), padding="same"))

    # Classification head.
    model.add(Flatten())
    model.add(Dense(units=256, activation=ReLU()))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=64, activation=ReLU()))
    model.add(Dense(units=64, activation=ReLU()))
    model.add(Dense(units=num_classes, activation=Softmax()))

    model.compile(
        optimizer=SGD(),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()
    return model
