# MangoLeafVarietyBD: Code & Baseline Evaluation 🥭

[![Dataset](https://img.shields.io/badge/Dataset-Mendeley_Data-blue)](https://doi.org/10.17632/hb3kvgfcvm.2)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c)](#)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](#)

This repository contains the official codebase for data preprocessing, background removal, offline augmentation, and baseline model evaluation for the **MangoLeafVarietyBD** dataset.

The dataset focuses on **8 distinct mango leaf varieties** from Bangladesh:
`Amrupali`, `Banana`, `Bandigori`, `Brunei King`, `Harivanga`, `Himsagor`, `Kacha Mitha`, and `Surjapuri`.

---

## 📂 Repository Structure & Workflow

To ensure high reproducibility and minimize computational overhead, the experimental workflow is modularized into two distinct phases:

### Phase 1: `01_Data_Preprocessing_and_Augmentation_Colab.ipynb`
* **Execution Environment:** Google Colab (utilizing Google Drive mounting).
* **Data Structuring:** The initial **2,744 raw images** are categorically labeled and organized into 8 class-specific directories.
* **Offline Augmentation (via Albumentations):** Applied to every raw image to expand the dataset **7-fold** (1 Original + 6 Augmented versions).
    * **Geometric:** Rotation (±15°, ±30°, ±45°), Horizontal/Vertical flips, and Shear (±10° - 20°).
    * **Photometric:** Brightness (0.6x to 1.4x) and Contrast (0.7x to 1.3x) adjustments.
    * **Noise:** Gaussian Blur (sigma 0.5 - 1.0) to simulate varying focus conditions.
* **Background Removal:** Rigorous pixel-level isolation of leaf morphology to eliminate field-induced contextual noise.
* **Normalization & Storage:** All images are resized to uniform resolution and exported directly to **Google Drive** for persistent storage.
* **Final Output:** A static augmented pool of **19,208 high-resolution images**.

### Phase 2: `02_Model_Training_and_Evaluation_Kaggle.ipynb`
* **Execution Environment:** Kaggle Cloud GPU (Hardware Accelerated).
* **Data Splitting:** Stratified split (Train: **80%**, Val: **10%**, Test: **10%**) using a fixed **Seed: 42** to maintain class ratios.
* **On-the-fly Training Augmentation:** Dynamic transformations applied via `torchvision.transforms` to enhance generalization:
    * `RandomResizedCrop(224, scale=(0.7, 1.0))`
    * `RandomHorizontalFlip(p=0.5)` & `RandomVerticalFlip(p=0.5)`
    * `RandomRotation(90°)`
    * `RandomPerspective(distortion_scale=0.4, p=0.5)`
    * `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`
    * `RandomGrayscale(p=0.1)`
    * `GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))`
    * `RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))`
* **Validation/Test Transforms:** Strictly limited to `Resize(224x224)` and `ToTensor()` to ensure unbiased evaluation on original features.
* **Image Normalization:** Global application of ImageNet constants: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

---

## ⚙️ Technical Details & Hyperparameters

The training engine utilizes the `timm` (PyTorch Image Models) library with the following exhaustive configurations:

### 🛠️ Global Parameters
* **Seed:** `42` (Enforced across PyTorch, NumPy, Random, and CuDNN deterministic algorithms).
* **Optimizer:** `AdamW` with architecture-specific weight decay.
* **LR Scheduler:** `CosineAnnealingLR` (T_max = 30) for smooth convergence.
* **Loss Function:** `CrossEntropyLoss` with **Dynamic Class Weighting** (`1. / np.bincount(train_labels)`) and label smoothing.
* **Computational Efficiency:** Mixed Precision Training using `torch.amp.autocast` and `GradScaler`.
* **Execution Strategy:** Batch Size: `32` | Max Epochs: `30` | Early Stopping: `7` (patience).

### 🏗️ Model-Specific Architecture Details
Each model utilizes `pretrained=True` (ImageNet-1k) weights with customized classification heads:

| Model | Dropout Rate | Learning Rate | Weight Decay | Label Smoothing |
| :--- | :---: | :---: | :---: | :---: |
| **ViT-B16** | 0.8 | 2e-4 | 0.50 | 0.50 |
| **DenseNet121** | 0.8 | 1.4e-3 | 0.50 | 0.50 |
| **GhostNetV2** | 0.8 | 5e-5 | 0.30 | 0.40 |
| **ResNet18** | 0.0 | 1e-4 | 0.00 | 0.00 |

---

## 📊 Evaluation Metrics & Performance

Performance is evaluated on a held-out test set (1,921 images) using a standard scientific reporting format (4 decimal places).

| Model | Architecture | Accuracy | Precision | Recall | F1-Score | Inference (ms/img) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ViT-B16** | Transformer | **97.55%** | **0.9762** | **0.9755** | **0.9756** | ~13.6 ms |
| **DenseNet121** | Dense CNN | 95.58% | 0.9565 | 0.9558 | 0.9558 | ~3.58 ms |
| **GhostNetV2** | Lightweight | 94.74% | 0.9482 | 0.9474 | 0.9474 | **~1.55 ms** |
| **ResNet18** | Residual CNN | 93.91% | 0.9417 | 0.9391 | 0.9394 | ~1.39 ms |

---

## 📈 Evaluation Artifacts
The `Evaluation_Results/` directory contains per-model sub-folders with:
* `Accuracy and Loss Curve.png`: Training vs Validation epoch-wise tracking.
* `Confusion Matrix.png`: Heatmap showing true vs predicted labels for all 8 varieties.
* `Classification Reports.txt`: Precision, Recall, and F1-score for each class (calculated with 4-digit precision).
* `training_logs.csv`: Raw data for all epochs (Training Loss/Acc, Validation Loss/Acc, Inference Time).

## 🚀 Reproduction Steps
1. **Download:** Get the raw images from [Mendeley Data Repository (V2)](https://doi.org/10.17632/hb3kvgfcvm.2).
2. **Preprocess:** Sequential run of `01_Data_Preprocessing...` in Google Colab.
3. **Train:** Run `02_Model_Training...` in Kaggle using the augmented images to regenerate all artifacts.

## 🛠️ Requirements
Python `3.12+`, PyTorch `2.9.0+cu126`, `timm` `1.0.24`, `Albumentations`, `Scikit-learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`.

---
*For inquiries regarding the methodology or dataset licensing, please refer to the corresponding data article: [MangoLeafVarietyBD (Mendeley Data)](https://doi.org/10.17632/hb3kvgfcvm.2)*
