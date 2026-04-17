# Image Classification with CNN

A small deep-learning project where we built and compared Convolutional Neural Networks to classify tiny images from the CIFAR-10 dataset into 10 categories.

[Task Descriptions and Project Instructions](https://github.com/sabeurzarai/Project-CNN-Group-2/tree/main/doc)

## Project Results

We trained **9 models** on CIFAR-10, starting simple and progressively adding techniques to see what moved the needle.

- **Baseline CNN** — one Conv layer, reaches \~55% test accuracy  
- **VGG-style architectures** — stacking conv blocks pushes us to 60–75%  
- **Training tricks** — Adam optimizer, BatchNorm, SGD with momentum and LR scheduling  
- **Regularization** — Dropout, L2 weight decay, data augmentation  
- **Transfer learning** — VGG16 pretrained on ImageNet, with a custom classification head

Our two stand-out models:

| Model | Approach | Test Accuracy | Train-Test Gap |
| :---- | :---- | :---- | :---- |
| **Model 8** | Custom VGG-style CNN with full regularization | **88.94%** | 7.64% |
| **Model 9** | Transfer learning from VGG16 | 87.05% | **2.75%** (best) |

Model 8 wins on raw accuracy. Model 9 generalizes much better — smaller gap, fewer trainable parameters, shorter training time. We deployed Model 7 in the live demo as a balance of both.

We deployed a simple Gradio demo where you can drop in an image or load a random photo per class to see the model in action.

## Repository Folders and Files

### Documents

**Group2 — Image Classification with CNN — Presentation Slides** Final slide deck used for the presentation.

**Group2 — Image Classification with CNN — Model Comparison** Full side-by-side comparison of all 9 models (architecture, training params, results).

### Notebook

**main.ipynb** — the full project in one notebook, covering:

- **Dataset exploration** — visualizes 10 samples per class, shows the 50k/10k split  
- **Preprocessing** — normalization to \[0, 1\], one-hot encoding, data augmentation  
- **Model 1** — baseline CNN: 1 Conv \+ Dense  
- **Model 2** — two stacked conv layers (VGG idea)  
- **Model 3** — full 3-block VGG-style, input resized to 64×64  
- **Model 4** — same architecture, but with Adam optimizer and smaller batch  
- **Model 5** — adds BatchNormalization after every Conv  
- **Model 6** — switches to SGD with momentum \+ LR scheduler, larger Dense layer  
- **Model 7** — adds Dropout, L2 regularization and data augmentation  
- **Model 8** — best custom CNN — tunes L2 and Dense size  
- **Model 9** — transfer learning from VGG16, two-stage fine-tuning  
- **Evaluation** — accuracy, precision, recall, F1, confusion matrices, gap analysis  
- **Deployment** — Gradio web app serving Model 7

Trained models are cached to Google Drive so the notebook loads them instead of retraining on each run.

### Additional folders

**cnn\_models/** — saved `.keras` files for each trained model (reused across sessions)

## Installation

Use **requirements.txt** to install the packages needed to run the notebook. A virtual environment is recommended.

python \-m venv .venv

.venv/Scripts/activate

pip install \-r requirements.txt

The notebook is designed for Google Colab with GPU enabled (**Runtime \> Change runtime type \> Hardware accelerator \> GPU**). It will also run locally, just slower.

## Team

Group 2 — Ironhack AI Bootcamp  
