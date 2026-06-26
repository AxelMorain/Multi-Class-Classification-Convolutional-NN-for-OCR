"""Image loading and preprocessing for the OCR character classifier.

The pipeline that reached 99% test accuracy is:
    raw PNG  ->  resize to 32x32  ->  per-image Otsu threshold  ->  (32, 32, 1)

Labels are read from the parent directory name (``.../training_data/A/xyz.png``
-> ``"A"``) rather than by indexing into the file-path string, so the code is
robust to path length and works the same on Windows, macOS and Linux.
"""

import glob
import os

import numpy as np
import skimage as ski
from PIL import Image

# 36 classes: digits 0-9 followed by the 26 uppercase letters.
CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
CLASS_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}

IMAGE_SIZE = 32  # all images are resized to IMAGE_SIZE x IMAGE_SIZE


def load_image_paths(data_dir: str) -> list:
    """Return a sorted list of every PNG under ``data_dir/<class>/*.png``.

    Args:
        data_dir: Directory containing one sub-folder per class
            (e.g. ``data/training_data``).
    Returns:
        Sorted list of file paths.
    """
    pattern = os.path.join(data_dir, "*", "*.png")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No PNG images found under {pattern!r}.")
    return paths


def label_from_path(path: str) -> str:
    """Extract the class label from an image path via its parent directory.

    ``.../training_data/A/00012.png`` -> ``"A"``.
    """
    return os.path.basename(os.path.dirname(path))


def load_dataset(data_dir: str) -> tuple:
    """Load every image under ``data_dir`` together with its label.

    Args:
        data_dir: Directory containing one sub-folder per class.
    Returns:
        ``(images, labels)`` where ``images`` is an object array of raw 2-D
        arrays (images may have differing dimensions) and ``labels`` is an
        array of single-character strings.
    """
    paths = load_image_paths(data_dir)
    images = np.array([np.array(Image.open(p)) for p in paths], dtype=object)
    labels = np.array([label_from_path(p) for p in paths])
    return images, labels


def preprocess_images(images: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize each image to a square and binarize it with per-image Otsu.

    Every image is resized to ``image_size x image_size`` then thresholded with
    its own Otsu threshold, producing high-contrast binary characters. The
    output is shaped ``(N, image_size, image_size, 1)`` ready for a Conv2D.

    Args:
        images: Iterable/array of 2-D grayscale image arrays.
        image_size: Side length to resize every image to.
    Returns:
        Float array of shape ``(N, image_size, image_size, 1)`` with values in
        ``{0.0, 1.0}``.
    """
    resized = np.array(
        [ski.transform.resize(im, (image_size, image_size)) for im in images]
    )
    binarized = np.array(
        [im >= ski.filters.threshold_otsu(im) for im in resized], dtype=np.float32
    )
    return binarized.reshape(-1, image_size, image_size, 1)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """One-hot encode string labels against the fixed :data:`CLASSES` order.

    Using a fixed, sorted class list (rather than fitting an encoder on the
    data) guarantees the same column ordering for train and test sets and makes
    the mapping trivially invertible with :func:`decode_labels`.

    Args:
        labels: Array of single-character class strings.
    Returns:
        Float array of shape ``(N, 36)``.
    """
    indices = np.array([CLASS_TO_INDEX[label] for label in labels])
    one_hot = np.zeros((len(labels), len(CLASSES)), dtype=np.float32)
    one_hot[np.arange(len(labels)), indices] = 1.0
    return one_hot


def decode_labels(one_hot: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_labels` — turn one-hot rows back into characters."""
    indices = np.argmax(one_hot, axis=1)
    return np.array([CLASSES[i] for i in indices])
