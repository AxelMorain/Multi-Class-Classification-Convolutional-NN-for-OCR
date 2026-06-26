# -*- coding: utf-8 -*-
"""
@author: Axel Morain
https://www.linkedin.com/in/axel-morain/

Experiment 02 — Model building, training and evaluation.

Notes / overview:
    Three vastly different architectures were tried during development;
    build_model_v4 stood out clearly as the best and is the one kept.

    Dropout-rate observations:
        - 0.25 : validation accuracy crosses 80% around epoch 13 (best).
        - 0.00 : ~18 epochs to cross 80%.
        - 0.30 : detrimental, >15 epochs to cross 80%.

    With the preprocessed + shuffled data, the model reaches ~99% test accuracy.
"""

import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import encode_labels, load_dataset, preprocess_images
from src.model import build_model_v4
from src.utils import plot_history

# Verify the GPU install of TF. If this prints nothing, TF is on the CPU.
print("GPUs:", tf.config.list_physical_devices("GPU"))


# Build the dataset ----------------------------------------------------------
test_images_raw, test_labels = load_dataset("data/testing_data")
train_images_raw, train_labels = load_dataset("data/training_data")

X_train = preprocess_images(train_images_raw)
X_test = preprocess_images(test_images_raw)
y_train = encode_labels(train_labels)
y_test = encode_labels(test_labels)

X_train, y_train = shuffle(X_train, y_train, random_state=5)
X_test, y_test = shuffle(X_test, y_test, random_state=5)


'''
-------------------------------------------------------------------------------
Quick architecture sanity run (30 epochs, no callbacks)
-------------------------------------------------------------------------------
Used during development to compare dropout rates and confirm the model learns.
'''
model = build_model_v4(dropout_rate=0.25)

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=30,
    validation_split=0.15,
    shuffle=True,
)

hh = history.history
print("history keys:", list(hh.keys()))
print("Validation acc:", hh["val_accuracy"])
plot_history(history, title="v4 — 30-epoch sanity run")


'''
-------------------------------------------------------------------------------
Full training run with callbacks
-------------------------------------------------------------------------------
Early stopping + checkpointing to keep the best model.
'''
early_stopping = EarlyStopping(monitor="val_loss", patience=10)
model_checkpoint = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
)

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=60,
    validation_split=0.15,
    shuffle=True,
    callbacks=[early_stopping, model_checkpoint],
)


'''
-------------------------------------------------------------------------------
Evaluate
-------------------------------------------------------------------------------
'''
train_metrics = model.evaluate(x=X_train, y=y_train)
test_metrics = model.evaluate(x=X_test, y=y_test)
print(
    "Test loss: {0}, test accuracy {1}".format(
        round(test_metrics[0], 2), round(test_metrics[1], 2)
    )
)
# ~99% test accuracy. Project goal achieved.

plt.figure()
plt.scatter(range(len(hh["val_accuracy"])), hh["val_accuracy"])
plt.title("Validation accuracy per epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy")
plt.show()
plt.clf()
