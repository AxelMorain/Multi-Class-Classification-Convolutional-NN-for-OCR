"""Display and reporting helpers for the OCR project."""

import inspect
import sys

import numpy as np
import matplotlib.pyplot as plt


def display_image(image, title: str = "An Image", cmap: str = "gray") -> None:
    """Show a single image with a title."""
    plt.imshow(np.squeeze(image), cmap=cmap)
    plt.title(str(title))
    plt.show()
    plt.clf()


def plot_history(history, title: str = "Training history") -> None:
    """Plot training loss and validation accuracy from a Keras History object."""
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.show()
    plt.clf()


def var_memory_report() -> None:
    """Print a memory-usage table for variables in the caller's global scope.

    Handy when the full image arrays are held in memory and you want to see
    which ones to ``del``.
    """
    caller_globals = inspect.stack()[1][0].f_globals

    total = 0
    rows = []
    for name, obj in sorted(caller_globals.items()):
        if name.startswith("_"):
            continue
        size = obj.nbytes if isinstance(obj, np.ndarray) else sys.getsizeof(obj)
        total += size
        rows.append((name, type(obj).__name__, size))

    rows.sort(key=lambda x: -x[2])
    print(f"{'Variable':30s} {'Type':20s} {'Size (MB)':>10}")
    print("-" * 65)
    for name, typ, size in rows:
        print(f"{name:30s} {typ:20s} {size / 1e6:>10.4f}")
    print("-" * 65)
    print(f"{'TOTAL':30s} {'':20s} {total / 1e6:>10.4f}")
