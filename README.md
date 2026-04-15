# Image Classification with CNNs — CIFAR-10

> Ironhack AI Bootcamp · Deep Learning Lab  
> A progressive study of Convolutional Neural Networks from baseline to transfer learning

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Model Progression](#model-progression)
- [Architecture Details](#architecture-details)
- [Training Strategy](#training-strategy)
- [Key Concepts](#key-concepts)
- [Results Interpretation](#results-interpretation)
- [File Structure](#file-structure)

---

## Overview

This lab explores image classification on CIFAR-10 through a series of 9 progressively improved CNN architectures. Each model introduces one or two new techniques, making it easy to isolate the contribution of each design decision — from a simple baseline CNN to a pretrained VGG16 with transfer learning.

The goal is not just to maximize accuracy, but to understand *why* each change helps and what tradeoffs it introduces.

---

## Dataset

**CIFAR-10** consists of 60,000 color images (32×32 pixels, 3 channels) across 10 classes:

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

- **Training set:** 50,000 images (5,000 per class)
- **Test set:** 10,000 images (1,000 per class)
- **Preprocessing:** pixel values normalized to [0, 1] by dividing by 255; labels one-hot encoded

---

## Environment Setup

```python
# Core libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and preprocess
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train_norm = x_train.astype('float32') / 255.0
x_test_norm  = x_test.astype('float32')  / 255.0
y_train_cat  = to_categorical(y_train, 10)
y_test_cat   = to_categorical(y_test, 10)
```

Models are saved to Google Drive to avoid retraining across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
MODEL_DIR = '/content/drive/MyDrive/cnn_models'
```

---

## Model Progression

Each model builds directly on the previous one by adding a single new concept:

```
Model 1  →  Baseline CNN (1 conv block)
Model 2  →  + 2 convs per block + padding=same
Model 3  →  + 3 blocks with growing filters (64→128→256) + input resize
Model 4  →  + Adam optimizer + EarlyStopping + smaller batch
Model 5  →  + BatchNormalization after every Conv layer
Model 6  →  + SGD with momentum + ReduceLROnPlateau + Dense(512)
Model 7  →  + Dropout(0.5) + Data Augmentation
Model 8  →  + L2 weight regularization on all layers
Model 9  →  Transfer Learning with pretrained VGG16 (2-stage training)
```

---

## Architecture Details

### Model 1 — Baseline CNN

The simplest possible architecture: one convolution, one pooling layer, one dense layer.

```
Input (32×32×3)
  → Conv2D(32 filters, 3×3, relu)       # output: 30×30×32
  → MaxPooling2D(2×2)                    # output: 15×15×32
  → Flatten                              # output: 7200
  → Dense(100, relu)
  → Dense(10, softmax)
```

**Compiled with:** SGD · `categorical_crossentropy` · 50 epochs · batch 512

---

### Model 2 — VGG-style Block

Inspired by the VGG paper (Simonyan & Zisserman, 2014). Stacking two conv layers before pooling lets the network compose features at the same resolution before downsampling. `padding='same'` preserves spatial dimensions after convolution.

```
Input (32×32×3)
  → Conv2D(32, 3×3, relu, same)         # output: 32×32×32
  → Conv2D(32, 3×3, relu, same)         # output: 32×32×32  ← NEW
  → MaxPooling2D(2×2)                    # output: 16×16×32
  → Flatten → Dense(128, relu) → Dense(10, softmax)
```

---

### Model 3 — Full VGG (3 Blocks)

Three blocks with progressively more filters capture features at different levels of abstraction. Input is upscaled from 32×32 to 64×64 to preserve spatial detail across the three pooling operations.

```
Input (32×32×3)
  → Resize(64×64)                        ← NEW
  → [Conv2D(64) × 2 → MaxPool]          ← low-level features
  → [Conv2D(128) × 2 → MaxPool]         ← mid-level features  ← NEW
  → [Conv2D(256) × 2 → MaxPool]         ← high-level features  ← NEW
  → Flatten → Dense(128, relu) → Dense(10, softmax)
```

**Compiled with:** SGD · 10 epochs (larger model, slower per epoch) · batch 512

---

### Model 4 — Adam + EarlyStopping

Same architecture as Model 3. Changes are entirely in the training strategy.

- **Adam** (`lr=0.003`): adaptive optimizer that adjusts the learning rate per parameter — faster convergence than plain SGD
- **`batch_size=128`**: smaller batches introduce more noise in gradient updates, often improving generalization
- **`EarlyStopping(patience=9, restore_best_weights=True)`**: halts training when `val_loss` stops improving for 9 epochs and reloads the best checkpoint

---

### Model 5 — BatchNormalization

`BatchNormalization` is added after every Conv layer. It normalizes the layer's activations to have zero mean and unit variance across the batch, then applies learnable scale and shift parameters.

```
Conv2D → BatchNormalization → Activation(relu) → ...
```

**Benefits:**
- Stabilizes training by reducing internal covariate shift
- Allows higher learning rates
- Acts as a mild regularizer, reducing the need for Dropout in some cases

---

### Model 6 — SGD with Momentum + LR Scheduler

Returns to SGD but with momentum and a learning rate schedule. The dense head is also enlarged.

```python
optimizer = SGD(learning_rate=0.01, momentum=0.9)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]
```

- **Momentum (0.9):** accumulates a velocity in consistent gradient directions, accelerating through flat regions and slowing in sharp curves
- **`ReduceLROnPlateau`:** halves the learning rate if `val_loss` doesn't improve for 5 consecutive epochs — allows coarse-to-fine optimization
- **`Dense(512)`:** larger head to leverage the richer features from BatchNorm + deeper architecture

---

### Model 7 — Dropout + Data Augmentation

Two anti-overfitting techniques added on top of Model 6.

**Dropout:**
```python
Dense(512, relu) → Dropout(0.5) → Dense(10, softmax)
```
During training, 50% of neurons are randomly disabled each batch, forcing the network to learn redundant representations. Disabled at inference time.

**Data Augmentation:**
```python
ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
```
Generates slightly modified versions of images on the fly each epoch — effectively multiplying the training set without storing extra data. The model never sees the exact same image twice.

---

### Model 8 — L2 Regularization

L2 (weight decay) adds a penalty term to the loss function proportional to the squared magnitude of all weights:

```
total_loss = cross_entropy_loss + λ × Σ(w²)
```

```python
REG = l2(1e-4)  # λ = 0.0001
Conv2D(64, ..., kernel_regularizer=REG)
Dense(256, ..., kernel_regularizer=REG)
```

Small weights are forced towards zero unless they genuinely reduce the task loss, discouraging the model from over-relying on any individual feature. Combined with Dropout and Augmentation, this is a three-layered defense against overfitting.

**Note:** Dense head is reduced from 512 to 256 neurons since L2 already handles part of the regularization load.

---

### Model 9 — Transfer Learning with VGG16

Instead of training from scratch, this model reuses VGG16 weights pretrained on ImageNet (1.2M images, 1000 classes).

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
```

**Architecture:**
```
Input (32×32×3)
  → Resize(64×64)
  → VGG16 feature extractor (frozen)
  → GlobalAveragePooling2D          ← replaces Flatten
  → Dense(256, relu)
  → Dropout(0.5)
  → Dense(10, softmax)
```

`GlobalAveragePooling2D` averages each feature map to a single value — more compact and less prone to overfitting than flattening.

**Two-stage training:**

| Stage | Layers trainable | Optimizer | LR | Epochs |
|-------|-----------------|-----------|-----|--------|
| 1 — Head only | Custom head | Adam | 0.001 | 20 |
| 2 — Fine-tuning | Head + top 8 VGG16 layers | SGD + momentum | 0.0001 | up to 12 |

In Stage 2, only the top 8 layers of VGG16 are unfrozen. The very low learning rate (0.0001) ensures the pretrained weights are nudged rather than overwritten. EarlyStopping and ReduceLROnPlateau are active throughout.

---

## Training Strategy

### Hyperparameter Summary

| Model | Optimizer | LR | Epochs | Batch | BN | Dropout | Aug | L2 |
|-------|-----------|----|--------|-------|----|---------|-----|----|
| 1 | SGD | default | 50 | 512 | ✗ | ✗ | ✗ | ✗ |
| 2 | SGD | default | 50 | 512 | ✗ | ✗ | ✗ | ✗ |
| 3 | SGD | default | 10 | 512 | ✗ | ✗ | ✗ | ✗ |
| 4 | Adam | 0.003 | 50* | 128 | ✗ | ✗ | ✗ | ✗ |
| 5 | Adam | 0.003 | 50* | 128 | ✓ | ✗ | ✗ | ✗ |
| 6 | SGD+mom | 0.01 | 50* | 128 | ✓ | ✗ | ✗ | ✗ |
| 7 | SGD+mom | 0.01 | 50* | 128 | ✓ | 0.5 | ✓ | ✗ |
| 8 | SGD+mom | 0.01 | 50* | 128 | ✓ | 0.5 | ✓ | 1e-4 |
| 9 | Adam→SGD | 0.001→0.0001 | 20+12* | 128 | ✓ (VGG) | 0.5 | ✓ | ✗ |

*\* = max epochs; EarlyStopping may terminate earlier*

### Loss Function

All models use **categorical cross-entropy**:

```
L = -Σ y_true × log(y_pred)
```

Measures the distance between the predicted probability distribution and the true one-hot label. Lower is better.

### Metrics

- **Accuracy:** fraction of correctly classified images
- **Confusion matrix:** reveals *which* classes get confused — crucial for understanding model weaknesses, not just overall performance

---

## Key Concepts

### Why activation functions?
Without non-linear activations, stacking layers collapses to a single linear transformation. ReLU (`max(0, x)`) is used in hidden layers for its simplicity and because it avoids the vanishing gradient problem of sigmoid/tanh.

### Sigmoid vs. Softmax
- **Sigmoid** outputs independent probabilities for each class — used for multi-label problems where multiple classes can be true simultaneously
- **Softmax** outputs a probability *distribution* that sums to 1 — used for multi-class problems where exactly one class is correct

### Binary vs. Categorical Cross-Entropy
- **Binary cross-entropy:** for two-class or multi-label problems (sigmoid output)
- **Categorical cross-entropy:** for multi-class problems (softmax output) — used throughout this lab

### BatchNorm vs. Dropout
| | BatchNorm | Dropout |
|--|-----------|---------|
| Acts on | Layer activations | Neurons |
| How | Normalizes distribution | Randomly disables |
| Active at test time | Yes | No (all neurons active) |
| Main benefit | Training stability | Prevents co-adaptation |

---

## Results Interpretation

### Confusion Matrix Analysis

Common confusions observed across models reflect genuine visual similarity in the dataset:

- **Cat ↔ Dog:** similar body shape, texture, and pose at 32×32 resolution
- **Automobile ↔ Truck:** shared structure (wheels, rectangular body, similar scale)
- **Deer ↔ Horse:** similar proportions and outdoor background context
- **Bird ↔ Airplane:** similar silhouettes — wings against sky backgrounds

Classes with distinct visual signatures (ship, frog) typically achieve higher per-class accuracy. Improvements across models should show reduced off-diagonal values in these hard pairs.

### What to look for in loss curves

| Pattern | Diagnosis |
|---------|-----------|
| Training loss ↓, validation loss ↑ | Overfitting |
| Both losses high and flat | Underfitting or too low LR |
| Both losses decreasing together | Healthy training |
| Val loss spikes then recovers | Batch noise — consider smaller LR or larger batch |
| Loss plateaus then drops | LR scheduler kicked in |

### Expected accuracy progression

Each architectural improvement should yield measurable gains. Transfer learning (Model 9) typically produces the largest jump because VGG16's ImageNet features are rich and general enough to transfer well to CIFAR-10 despite the domain difference.

---

## File Structure

```
.
├── main_9_models.ipynb     # Main notebook with all 9 models
├── README.md               # This file
└── cnn_models_summary.html # Visual architecture overview
```

Models are persisted to Google Drive under `MyDrive/cnn_models/` as `.keras` files and reloaded automatically on subsequent runs to avoid redundant training.

---

*Ironhack AI Bootcamp — Deep Learning module*
