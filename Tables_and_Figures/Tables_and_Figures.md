# Tables and Figures for the MangoLeafVarietyBD Dataset

This document contains all tables and figures used in the manuscript.

---

## Table 1: Comparison of MangoLeafVarietyBD with Existing Bangladeshi Datasets

| Mango Dataset | Number of Classes | Total Raw Images | Resolution | Geographical Coverage |
| :------------ | :--------------- | :-------------- | :-------- | :------------------ |
| BDMANGO [2] | 6 | 837 | 640 x 480 px | Dhaka, Bangladesh |
| MangoLeafVarietyBD | 8 | 2,744 | Up to 4000 x 3000 px | Thakurgaon & Savar, Bangladesh |

---

## Table 2: Quick Overview of MangoLeafVarietyBD

| Data Type | Raw field-captured mango leaf images (original resolution; no resizing applied) |
| :-------- | :--------------------------------------------------------- |
| Data Format | Images are JPG |
| Total Images | 2,744 raw images |
| Number of Classes/Varieties | Amrupali, Banana, Bandigori, Himsagar, Surjapuri, Harivanga, Kacha Mitha, Brunei King |
| Distribution of Original Images | Amrupali: 308, Banana: 351, Bandigori: 332, Himsagar: 341, Surjapuri: 370, Harivanga: 366, Kacha Mitha: 376, Brunei King: 300 |
| Acquisition Method | Snapping on white background using two mobile phones; both sides of leaf captured |
| Data Source Locations | Thakurgaon District and Daffodil International University, Savar, Dhaka |
| Where Applicable | Classification, image classification, phenotyping, agricultural ML studies |

---

## Table 3: Architecture-Specific Hyperparameters Used for Model Training

| Model | Learning Rate | Dropout | Weight Decay | Label Smoothing |
| :---- | :-----------: | :----: | :----------: | :------------: |
| ViT-B16 | 2×10⁻⁴ | 0.8 | 0.50 | 0.50 |
| DenseNet121 | 1.4×10⁻³ | 0.8 | 0.50 | 0.50 |
| GhostNetV2 | 5×10⁻⁵ | 0.8 | 0.30 | 0.40 |
| ResNet18 | 1×10⁻⁴ | 0.0 | 0.00 | 0.00 |

---

## Table 4: Benchmark Performance of Deep Learning Models on the MangoLeafVarietyBD Test Set (N = 1,921)

| Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Latency (ms) | Throughput (FPS) |
| :---- | :----------: | :----------------: | :---------------: | :----------------: | :-----------: | :-------------: |
| ViT-B16 | 0.9755 | 0.9762 | 0.9755 | 0.9756 | 13.6739 | ~73.13 |
| DenseNet121 | 0.9558 | 0.9565 | 0.9558 | 0.9558 | 3.5861 | ~278.85 |
| GhostNetV2 | 0.9474 | 0.9482 | 0.9474 | 0.9474 | 1.5566 | ~642.42 |
| ResNet18 | 0.9391 | 0.9417 | 0.9391 | 0.9394 | 1.4083 | ~710.07 |

---

## Fig. 1: Examples of Eight Distinct Types of Original Mango Leaf Images

![Fig1](Tables_and_Figures/Fig. 1. Examples of eight distinct types of original mango leaf images included in the dataset.png)

---

## Fig. 2: Structure of the Directory of the Dataset

![Fig2](Tables_and_Figures/Fig. 2. Structure of the directory of the dataset.png)

---

## Fig. 3: Flowchart Showing the Data Preparation Steps

![Fig3](Tables_and_Figures/Fig. 3. Flowchart showing the data preparation steps.png)

---

## Fig. 4: Sample of Augmented Images of MangoLeafVarietyBD Dataset (for Experimental Model Training)

![Fig4](Tables_and_Figures/Fig. 4. Sample of augmented images of MangoLeafVarietyBD Dataset (for Experimental Model Training).png)

---

## Fig. 5: Confusion Matrices of Four Deep Learning Models on MangoLeafVarietyBD Test Set (N=1,921)

| (a) ViT-B16 | (b) DenseNet121 |
| :---------: | :-------------: |
| ![Fig5a](fig5a_confusion_matrix_vit.png) | ![Fig5b](fig5b_confusion_matrix_densenet.png) |

| (c) GhostNetV2 | (d) ResNet18 |
| :-------------: | :-----------: |
| ![Fig5c](fig5c_confusion_matrix_ghostnet.png) | ![Fig5d](fig5d_confusion_matrix_resnet.png) |

---

## Fig. 6: Training Dynamics (Loss and Accuracy Curves) of Four Deep Learning Models

| (a) ViT-B16 | (b) DenseNet121 |
| :---------: | :-------------: |
| ![Fig6a](fig6a_training_curves_vit.png) | ![Fig6b](fig6b_training_curves_densenet.png) |

| (c) GhostNetV2 | (d) ResNet18 |
| :-------------: | :-----------: |
| ![Fig6c](fig6c_training_curves_ghostnet.png) | ![Fig6d](fig6d_training_curves_resnet.png) |
