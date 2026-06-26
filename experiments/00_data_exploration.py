# -*- coding: utf-8 -*-
"""
@author: Axel Morain
https://www.linkedin.com/in/axel-morain/

Experiment 00 — Data import and exploration.

Project:
Multi-class classification CNN for Optical Character Recognition. 36 classes:
digits 0-9 and the 26 uppercase letters of the alphabet.

Dataset:
Kaggle — Standard OCR Dataset
https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

Notes / overview:
    Manual exploration showed that not all images share the same dimensions, so
    the distribution of image widths and heights is plotted below. The
    distribution is roughly normal (not bimodal), so resizing every image to a
    fixed square is good enough — a bimodal distribution would have required
    more careful handling.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import load_dataset
from src.utils import display_image


'''
-------------------------------------------------------------------------------
Importing pictures
-------------------------------------------------------------------------------
'''
print(os.getcwd())

test_images_raw, test_labels = load_dataset("data/testing_data")
train_images_raw, train_labels = load_dataset("data/training_data")

print("test image[0] shape :", test_images_raw[0].shape)
print("train image[0] shape:", train_images_raw[0].shape)
print("first 20 train labels:", train_labels[:20])


'''
-------------------------------------------------------------------------------
Image dimensions
-------------------------------------------------------------------------------
From manual exploration, not all images have the same dimensions. Let's have a
closer look at the distribution of widths and heights.
'''

train_dim_x = [im.shape[1] for im in train_images_raw]
train_dim_y = [im.shape[0] for im in train_images_raw]

plt.hist(train_dim_x, bins=100, color="b", alpha=0.6)
plt.hist(train_dim_y, bins=100, color="r", alpha=0.6)
plt.title("Distribution of training image dimensions")
plt.legend(("X axis length", "Y axis length"))
plt.show()
plt.clf()

# The distribution looks fairly normal, so resizing every image using a fixed
# square is good enough. A bimodal distribution would have required more work.


'''
-------------------------------------------------------------------------------
Brightness / contrast
-------------------------------------------------------------------------------
The images are already grayscale. Let's look at the brightness range of a few
of them.
'''

for i in range(0, len(train_images_raw), max(1, len(train_images_raw) // 10)):
    plt.hist(np.array(train_images_raw[i]).ravel(), bins=256)
    plt.title("Histogram of train image #" + str(i))
    plt.show()
    plt.clf()

# The histograms show a very high brightness range with a sharp transition
# between bright and dark pixels: high-contrast images, which is exactly what we
# want. After scrolling through ~80 images by hand, no denoising or smoothing is
# needed — thresholding (see experiment 01) will be enough.

display_image(test_images_raw[0], title="A raw test image")
