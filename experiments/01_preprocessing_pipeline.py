# -*- coding: utf-8 -*-
"""
@author: Axel Morain
https://www.linkedin.com/in/axel-morain/

Experiment 01 — Preprocessing pipeline.

Notes / overview:
    Two ideas were explored to help the model learn:

    1. Resizing images to squares (32x32). All the standard pre-trained models
       take square inputs, so it seemed worth trying. On its own this did NOT
       improve learning, but it was kept because it standardizes the input shape.

    2. Per-image Otsu thresholding. Because the images are high-contrast, an
       Otsu threshold produces very sharp binary characters. This worked great.

    The single change that actually unlocked learning was shuffling the data
    before training (see the bottom of this file and experiment 02).
"""

import os
import sys

import numpy as np
import skimage as ski

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from sklearn.utils import shuffle

from src.preprocessing import encode_labels, load_dataset, preprocess_images
from src.utils import display_image


# Load raw data --------------------------------------------------------------
test_images_raw, test_labels = load_dataset("data/testing_data")
train_images_raw, train_labels = load_dataset("data/training_data")


'''
-------------------------------------------------------------------------------
Resizing to 32x32
-------------------------------------------------------------------------------
'''
IMAGE_SIZE = 32

sample = np.array(train_images_raw[0])
print("test shape:", sample.shape)
display_image(sample, title="Before resizing")

sample_resized = ski.transform.resize(sample, (IMAGE_SIZE, IMAGE_SIZE))
display_image(sample_resized, title=f"After resizing: {IMAGE_SIZE}x{IMAGE_SIZE}")


'''
-------------------------------------------------------------------------------
Otsu thresholding
-------------------------------------------------------------------------------
Pick a few interesting letters and threshold them to confirm the output is
sharp.
'''
for idx in (1500, 8000, 15000):
    if idx < len(train_images_raw):
        im = ski.transform.resize(
            np.array(train_images_raw[idx]), (IMAGE_SIZE, IMAGE_SIZE)
        )
        binary = im >= ski.filters.threshold_otsu(im)
        display_image(binary, title=f"Otsu threshold — train #{idx}")
# Super sharp! Thresholding alone is enough.


'''
-------------------------------------------------------------------------------
Apply the full pipeline
-------------------------------------------------------------------------------
preprocess_images() resizes + thresholds + reshapes to (N, 32, 32, 1).
'''
X_train = preprocess_images(train_images_raw)
X_test = preprocess_images(test_images_raw)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# Sanity check a few samples.
for idx in (1500, 15000):
    if idx < len(X_train):
        display_image(X_train[idx], title=f"Preprocessed train #{idx}")


'''
-------------------------------------------------------------------------------
Label encoding
-------------------------------------------------------------------------------
One-hot encode the 36 classes against a fixed class order so the train and test
columns line up and the mapping stays invertible.
'''
y_train = encode_labels(train_labels)
y_test = encode_labels(test_labels)
print("y_train shape:", y_train.shape)


'''
-------------------------------------------------------------------------------
Shuffle — the key step
-------------------------------------------------------------------------------
Shuffling the arrays before training is what allowed the model to learn. Without
it, validation accuracy stalls around 0.074 no matter the architecture or
hyper-parameters. ``shuffle=True`` inside ``model.fit`` did NOT substitute for
this.
'''
X_train, y_train = shuffle(X_train, y_train, random_state=5)
X_test, y_test = shuffle(X_test, y_test, random_state=5)

for i in range(5):
    display_image(X_train[i], title=str(y_train[i]))
