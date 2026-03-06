# MangoLeafVarietyBD: Code & Baseline Evaluation 🥭

[![Dataset](https://img.shields.io/badge/Dataset-Mendeley_Data-blue)](https://doi.org/10.17632/hb3kvgfcvm.2)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c)](#)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](#)

This repository contains the official codebase for data preprocessing, augmentation, and baseline model evaluation associated with the **MangoLeafVarietyBD** dataset. 

The dataset comprises high-resolution images of **8 distinct mango leaf varieties** collected in Bangladesh:
`Amrupali`, `Banana`, `Bandigori`, `Brunei King`, `Harivanga`, `Himsagor`, `Kacha Mitha`, and `Surjapuri`.

---

## 📂 Repository Structure & Workflow

To ensure high reproducibility and resource efficiency, the experimental workflow is modularized into two distinct environments:

### 1. `01_Data_Preprocessing_and_Augmentation_Colab.ipynb`
* **Execution Environment:** Google Colab
* **Purpose:** Handles initial data parsing, background removal, and dynamic data augmentation.
* **Applied Augmentations (on-the-fly):** `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation (90°)`, `RandomPerspective`, `ColorJitter`, `RandomGrayscale`, `GaussianBlur`, and `RandomErasing`.
* **Output:** Expands the raw dataset from **2,744 original images** to a robust training pool of **19,208 augmented images**.

### 2. `02_Model_Training_and_Evaluation_Kaggle.ipynb`
* **Execution Environment:** Kaggle Cloud GPU (Hardware Accelerated)
* **Purpose:** Trains and rigorously evaluates four state-of-the-art deep learning baseline models using a stratified split (Train: 80%, Val: 10%, Test: 10%).

---

## ⚙️ Hyperparameters & Training Configuration

To guarantee exact reproducibility, the specific training configurations used for the baseline models are documented below:

* **Global Settings:** * Image Size: `224 x 224`
  * Batch Size: `32`
  * Max Epochs: `30`
  * Early Stopping Patience: `7`
  * Optimizer: `AdamW`
  * LR Scheduler: `CosineAnnealingLR`
  * Precision: `Mixed Precision (torch.amp.autocast)`

* **Model-Specific Configurations:**
  * **ViT-B16:** Learning Rate: `2e-4`, Weight Decay: `0.50`, Label Smoothing: `0.50`
  * **DenseNet121:** Learning Rate: `1.4e-3`, Weight Decay: `0.50`, Label Smoothing: `0.50`
  * **GhostNetV2:** Learning Rate: `5e-5`, Weight Decay: `0.30`, Label Smoothing: `0.40`
  * **ResNet18:** Learning Rate: `1e-4`, Weight Decay: `0.0`, Label Smoothing: `0.0`

---

## 📊 Baseline Model Performance

The evaluation demonstrates strong discriminative capabilities across all eight mango leaf varieties. The performance metrics on the test set are summarized below:

| Model | Architecture Type | Accuracy | Precision | Recall | F1-Score | Inference Time (ms/step) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ViT-B16** | Vision Transformer | **97.55%** | **0.9762** | **0.9755** | **0.9756** | ~13.6 ms |
| **DenseNet121** | Dense CNN | 95.58% | 0.9565 | 0.9558 | 0.9558 | ~3.58 ms |
| **GhostNetV2** | Lightweight CNN | 94.74% | 0.9482 | 0.9474 | 0.9474 | **~1.55 ms** |
| **ResNet18** | Foundational CNN | 93.91% | 0.9417 | 0.9391 | 0.9394 | ~1.39 ms |

*(Note: GhostNetV2 proves highly viable for resource-constrained mobile deployments in agricultural fields due to its ultra-fast inference speed.)*

---

## 🚀 How to Reproduce

1. **Obtain the Dataset:** Download the raw image dataset from the official [Mendeley Data Repository (V2)](https://doi.org/10.17632/hb3kvgfcvm.2).
2. **Preprocess:** Open `01_Data_Preprocessing...` in Google Colab. Mount your drive, point the paths to the downloaded raw dataset, and run all cells to generate the augmented images.
3. **Train Models:** Upload the generated augmented dataset to a Kaggle environment. Execute `02_Model_Training...` to reproduce the model weights, accuracy metrics, confusion matrices, and loss curves.

## 📈 Evaluation Results
The generated visualizations, including the **Confusion Matrices** and **Loss/Accuracy Curves** for the evaluated models, can be found in the `Evaluation_Results` directory of this repository.

## 🛠️ Dependencies & Environment
* Python `3.12+`
* PyTorch `2.9.0`
* PyTorch Image Models (`timm`) `1.0.24`
* Albumentations, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---
*For any questions regarding the dataset or the baseline implementations, please refer to the corresponding data article: [MangoLeafVarietyBD (Mendeley Data)](https://doi.org/10.17632/hb3kvgfcvm.2)*
