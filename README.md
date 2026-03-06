# MangoLeafVarietyBD: Code & Baseline Evaluation 🥭

[![Dataset](https://img.shields.io/badge/Dataset-Mendeley_Data-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-ee4c2c)](#)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey)](#)

This repository contains the official codebase for data preprocessing, augmentation, and baseline model evaluation associated with the **MangoLeafVarietyBD** dataset. The dataset comprises high-resolution images of eight distinct mango leaf varieties collected from diverse geographical regions in Bangladesh (Thakurgaon and Savar) to facilitate robust computer vision applications in precision agriculture.

## 📂 Repository Structure & Workflow

To ensure high reproducibility and resource efficiency, the experimental workflow is modularized into two distinct environments:

### 1. `01_Data_Preprocessing_and_Augmentation_Colab.ipynb`
* **Execution Environment:** Google Colab
* **Purpose:** Handles initial data parsing, background removal, and dynamic data augmentation.
* **Details:** Utilizes the `Albumentations` library to simulate real-world field variability (e.g., random perspective, color jitter, gaussian blur, and random erasing).
* **Output:** Expands the raw dataset from **2,744 original images** to a robust training pool of **19,208 augmented images**.

### 2. `02_Model_Training_and_Evaluation_Kaggle.ipynb`
* **Execution Environment:** Kaggle Cloud GPU (Hardware Accelerated)
* **Purpose:** Trains and rigorously evaluates four state-of-the-art deep learning baseline models on the augmented dataset using a 80-10-10 stratified split.
* **Techniques:** Implements Cosine Annealing Learning Rate, Label Smoothing, and Mixed Precision (`torch.amp.autocast`) for optimal convergence.

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

1. **Obtain the Dataset:** Download the raw image dataset from the official [Mendeley Data Repository](#) *(Link will be updated upon publication)*.
2. **Preprocess:** Open `01_Data_Preprocessing_and_Augmentation_Colab.ipynb` in Google Colab. Mount your drive, point the paths to the downloaded raw dataset, and run all cells to generate the augmented images.
3. **Train Models:** Upload the generated augmented dataset to a Kaggle environment. Execute `02_Model_Training_and_Evaluation_Kaggle.ipynb` to reproduce the model weights, accuracy metrics, confusion matrices, and loss curves.

## 📈 Evaluation Results
The generated visualizations, including the **Confusion Matrices** and **Loss/Accuracy Curves** for the evaluated models, can be found in the `Evaluation_Results` directory of this repository.

## 🛠️ Dependencies & Environment

The codebase strictly adheres to the following dependencies to ensure reproducibility:
* Python `3.12+`
* PyTorch `2.9.0`
* PyTorch Image Models (`timm`) `1.0.24`
* Albumentations
* Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---
*For any questions regarding the dataset or the baseline implementations, please refer to the corresponding data article.*
