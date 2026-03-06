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

To ensure absolute reproducibility and resource efficiency, the experimental workflow is modularized into two distinct phases, executed in the exact sequence below:

### 1. `01_Data_Preprocessing_and_Augmentation_Colab.ipynb`
* **Execution Environment:** Google Colab
* **Data Labeling & Structuring:** The initial **2,744 raw images** are parsed and categorically labeled by organizing them into 8 distinct folders named after their respective classes.
* **Offline Augmentation:** Before any pixel alteration, the raw images are augmented to simulate natural field variability. 6 synthetic versions are generated for every original image using deterministic transformations (Rotation, Flipping, Brightness/Contrast, Shear, and Gaussian Blur). This expands the dataset to a total of **19,208 images**.
* **Background Removal:** Each of the 19,208 images then undergoes a rigorous background removal process to isolate the leaf area, ensuring the models focus purely on leaf morphology and minimizing contextual bias.
* **Resizing:** The isolated leaf images are subsequently resized to standard uniform dimensions to optimize computational efficiency.
* **Storage (Google Drive):** The final preprocessed and augmented dataset is seamlessly exported and saved directly to **Google Drive** for persistent storage and easy transfer to the training environment.

### 2. `02_Model_Training_and_Evaluation_Kaggle.ipynb`
* **Execution Environment:** Kaggle Cloud GPU (Hardware Accelerated)
* **Purpose:** Trains and rigorously evaluates four state-of-the-art deep learning baseline models.
* **Data Splitting:** A stratified split is applied to ensure uniform class distribution across sets -> **Train: 80%**, **Validation: 10%**, **Test: 10%** (Random State/Seed: `42`).
* **On-the-fly Augmentation (Training Set Only):** Dynamic real-time augmentations via `torchvision.transforms` within the PyTorch DataLoader to prevent overfitting during training:
  * `RandomResizedCrop(224, scale=(0.7, 1.0))`
  * `RandomHorizontalFlip(p=0.5)` & `RandomVerticalFlip(p=0.5)`
  * `RandomRotation(degrees=90)`
  * `RandomPerspective(distortion_scale=0.4, p=0.5)`
  * `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`
  * `RandomGrayscale(p=0.1)`
  * `GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))`
  * `RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))`
* **Validation/Test Transforms:** Strictly limited to `Resize(224x224)` and `ToTensor()` to evaluate the models on unaltered baseline conditions.
* **Normalization:** ImageNet standards `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` applied globally to all splits.

---

## ⚙️ Hyperparameters & Training Configuration

To guarantee exact reproducibility, the `timm` library was utilized with the following exhaustive configurations:

### Global Configuration
* **Seed:** `42` (Enforced across PyTorch, NumPy, Random, and CuDNN deterministic algorithms)
* **Input Resolution:** `224 x 224`
* **Batch Size:** `32` (with `num_workers=4`)
* **Max Epochs:** `30` (with Early Stopping Patience: `7`)
* **Optimizer & Scheduler:** `AdamW` paired with `CosineAnnealingLR (T_max=30)`
* **Loss Function:** `CrossEntropyLoss` utilizing **Dynamic Class Weighting** (`1. / np.bincount(train_labels)`) to mitigate natural class imbalances.
* **Precision:** Mixed Precision (`torch.amp.autocast` & `GradScaler`)
* **Pretrained Weights:** Enabled (`pretrained=True`) for all architectures.

### Model-Specific Configurations
Custom dense/classifier heads were added with specific dropout rates to prevent overfitting:

* **ViT-B16:** * Dropout: `0.8` | Learning Rate: `2e-4` | Weight Decay: `0.50` | Label Smoothing: `0.50`
* **DenseNet121:** * Dropout: `0.8` | Learning Rate: `1.4e-3` | Weight Decay: `0.50` | Label Smoothing: `0.50`
* **GhostNetV2:** * Dropout: `0.8` | Learning Rate: `5e-5` | Weight Decay: `0.30` | Label Smoothing: `0.40`
* **ResNet18:** * Dropout: `0.0` | Learning Rate: `1e-4` | Weight Decay: `0.0` | Label Smoothing: `0.0`

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
2. **Preprocess:** Open `01_Data_Preprocessing...` in Google Colab. Mount your drive, point the paths to the downloaded raw dataset, and run all cells sequentially to label, augment, remove backgrounds, resize, and save the final images to your Drive.
3. **Train Models:** Upload the generated augmented dataset from your Drive to a Kaggle environment. Execute `02_Model_Training...`. 
4. **Outputs:** The script automatically generates and saves the following artifacts for each model in the `All_outputs` directory:
   * `best_model.pth` (Model weights)
   * `training_logs.csv` (Epoch-wise loss/acc tracking)
   * `report.txt` (Scikit-learn classification report)
   * `cm.png` & `metrics.png` (Visualizations)

## 🛠️ Dependencies & Environment
* Python `3.12+`
* PyTorch `2.9.0+cu126`
* PyTorch Image Models (`timm`) `1.0.24`
* Albumentations, Scikit-learn, Pandas, NumPy, Pillow, Matplotlib, Seaborn

---
*For any questions regarding the dataset or the baseline implementations, please refer to the corresponding data article: [MangoLeafVarietyBD (Mendeley Data)](https://doi.org/10.17632/hb3kvgfcvm.2)*
