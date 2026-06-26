# Multi-Class OCR — Character Classification with a Convolutional NN

A convolutional neural network that classifies images of typed characters across
many fonts into **36 classes** (digits `0-9` and the uppercase letters `A-Z`),
achieving **~99% test accuracy**.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **~99%** |
| Classes | 36 (0-9, A-Z) |
| Input size | 32 × 32 × 1 |

![Training history](models/training_history.png)

---

## Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/AxelMorain/Multi-Class-Classification-Convolutional-NN-for-OCR.git
cd Multi-Class-Classification-Convolutional-NN-for-OCR
```

**2. Download the dataset**

Download the **Standard OCR Dataset** from Kaggle and arrange it so the class
sub-folders sit under `data/`:
> https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

```
data/
├── training_data/
│   ├── 0/  1/  ...  9/  A/  B/  ...  Z/
└── testing_data/
    ├── 0/  1/  ...  9/  A/  B/  ...  Z/
```

**3. Create the environment**
```bash
conda env create -f environment.yml
conda activate ocr-classification
```

**4. Run the full pipeline**
```bash
python main.py
```

This loads the raw images, preprocesses them, trains the model, evaluates on the
test set, and saves the best model to `models/`.

---

## Project Structure

```
├── src/                        # Shared, reusable modules
│   ├── preprocessing.py        # Image loading, resize + Otsu, label encoding
│   ├── model.py                # CNN architecture (build_model_v4)
│   └── utils.py                # Display and reporting helpers
│
├── experiments/                # Numbered scripts — the full R&D journey
│   ├── 00_data_exploration.py      # Image dimensions + brightness exploration
│   ├── 01_preprocessing_pipeline.py# Resize, Otsu, label encoding, the shuffle fix
│   └── 02_model_training.py        # Architecture, dropout study → ~99% accuracy
│
├── models/                     # Saved model weights and performance plots
├── data/                       # Raw images placeholder (not committed)
│
├── main.py                     # Clean end-to-end training entry point
├── environment.yml             # Conda environment (pins CUDA + all deps)
└── requirements.txt            # Pip-only alternative
```

---

## How It Works

### Step 1 — Preprocessing

Each raw character image is:

1. **Resized** to a fixed `32 × 32` square. Image dimensions vary slightly across
   the dataset; the distribution is roughly normal, so a single resize is enough.
2. **Binarized** with a per-image **Otsu threshold**. The images are
   high-contrast to begin with, so Otsu produces very sharp black-and-white
   characters with no denoising required.
3. **Reshaped** to `(N, 32, 32, 1)` for the Conv2D input.

Labels are read from each image's **parent directory name**
(`.../training_data/A/xyz.png` → `"A"`) and one-hot encoded against a fixed,
sorted class list so the train/test columns always align.

### Step 2 — The shuffle that made everything work

The single most important fix in this project: **shuffle the data before
training.** Without an explicit shuffle, validation accuracy stalls around
`0.074` regardless of architecture or hyper-parameters. Passing `shuffle=True` to
`model.fit` is **not** a substitute — shuffling the arrays up front is what let
the model train past `0.90`.

### Step 3 — Model Architecture

`build_model_v4` was selected after trying three vastly different architectures.

```
Input (32×32×1)
│
├─ MaxPooling2D (2×2)
├─ Conv2D (25 filters, 5×5)  →  ReLU  →  MaxPool (2×2)
├─ Conv2D (25 filters, 3×3)  →  ReLU  →  MaxPool (2×2)
├─ Conv2D (25 filters, 3×3)  →  ReLU  →  MaxPool (2×2)
├─ Conv2D (25 filters, 3×3)  →  ReLU  →  MaxPool (2×2)
├─ Conv2D (25 filters, 3×3)  →  ReLU  →  MaxPool (2×2)
├─ Flatten
├─ Dense (256)  →  ReLU  →  Dropout (0.25)
├─ Dense (64)   →  ReLU
├─ Dense (64)   →  ReLU
└─ Dense (36)   →  Softmax
```

**Optimizer:** SGD  
**Loss:** CategoricalCrossentropy  
**Callbacks:** ModelCheckpoint + EarlyStopping (patience=10)

The dropout rate visibly affects convergence speed: at `0.25` the model crosses
80% validation accuracy around epoch 13; with no dropout it takes ~18 epochs;
raising it to `0.30` becomes detrimental.

---

## Dataset

**Standard OCR Dataset** — Kaggle. 36 classes (digits + uppercase letters).

> https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

Raw images and saved model weights are not committed to this repo due to size.
Follow the Quick Start to regenerate them.

---

## Environment

TensorFlow 2.10.0 is the **last version with native Windows GPU support.**
Upgrading TensorFlow on Windows will disable GPU training.

```bash
conda env create -f environment.yml   # recommended — handles CUDA automatically
conda activate ocr-classification

# or, for CPU / cloud environments:
pip install -r requirements.txt
```
