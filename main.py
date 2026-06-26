# -*- coding: utf-8 -*-
"""
Multi-Class OCR Character Classification — full training pipeline.

Reproduces the ~99% test accuracy result end-to-end:
    1. Load raw character images from data/training_data/ and data/testing_data/
    2. Preprocess (resize to 32x32 + per-image Otsu threshold)
    3. One-hot encode the 36 class labels
    4. Shuffle  (critical — see note below)
    5. Train build_model_v4 with early stopping + checkpointing
    6. Evaluate on the held-out test set
    7. Plot training history and save the trained model to models/

Dataset:
    Kaggle — Standard OCR Dataset
    https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset
    36 classes: digits 0-9 and the 26 uppercase letters.

Note on shuffling:
    Passing ``shuffle=True`` to ``model.fit`` is NOT a substitute for shuffling
    the arrays beforehand. Without an explicit shuffle the model stalls around
    0.074 validation accuracy; shuffling the data first lets it train past 0.90.

Usage:
    conda activate ocr-classification
    python main.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Path setup -----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import encode_labels, load_dataset, preprocess_images
from src.model import build_model_v4

# --- GPU setup -------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU detected: {gpus[0].name}")
else:
    print("No GPU detected — running on CPU.")


'''
-------------------------------------------------------------------------------
1. Load raw images
-------------------------------------------------------------------------------
'''

print("\nLoading images...")

train_images_raw, train_labels_raw = load_dataset("data/training_data")
test_images_raw, test_labels_raw = load_dataset("data/testing_data")

print(f"  Training images loaded : {len(train_images_raw)}")
print(f"  Testing images loaded  : {len(test_images_raw)}")


'''
-------------------------------------------------------------------------------
2. Preprocess  (resize to 32x32 + per-image Otsu threshold)
-------------------------------------------------------------------------------
'''

print("\nPreprocessing...")

X_train = preprocess_images(train_images_raw)
X_test = preprocess_images(test_images_raw)

del train_images_raw, test_images_raw


'''
-------------------------------------------------------------------------------
3. One-hot encode the labels
-------------------------------------------------------------------------------
'''

y_train = encode_labels(train_labels_raw)
y_test = encode_labels(test_labels_raw)


'''
-------------------------------------------------------------------------------
4. Shuffle  (critical — see module docstring)
-------------------------------------------------------------------------------
'''

X_train, y_train = shuffle(X_train, y_train, random_state=5)
X_test, y_test = shuffle(X_test, y_test, random_state=5)

print(f"\nDataset:")
print(f"  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)} samples")


'''
-------------------------------------------------------------------------------
5. Train
-------------------------------------------------------------------------------
'''

print("\nBuilding model...")
model = build_model_v4(dropout_rate=0.25)

checkpoint_cb = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stop_cb = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

print("\nTraining...")
history = model.fit(
    x=X_train,
    y=y_train,
    epochs=60,
    validation_split=0.15,
    shuffle=True,
    callbacks=[checkpoint_cb, early_stop_cb],
)


'''
-------------------------------------------------------------------------------
6. Evaluate
-------------------------------------------------------------------------------
'''

print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n  Test loss     : {test_loss:.4f}")
print(f"  Test accuracy : {test_accuracy * 100:.2f}%")


'''
-------------------------------------------------------------------------------
7. Plot training history
-------------------------------------------------------------------------------
'''

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="validation")
axes[0].set_title("Loss (CategoricalCrossentropy)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train acc")
axes[1].plot(history.history["val_accuracy"], label="val acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle(f"OCR CNN — Final result: {test_accuracy * 100:.2f}% test accuracy")
plt.tight_layout()
plt.savefig("models/training_history.png", dpi=150)
plt.show()

print("\nDone. Model saved to models/best_model.keras")
