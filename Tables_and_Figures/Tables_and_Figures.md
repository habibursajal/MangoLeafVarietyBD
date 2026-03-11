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

![Fig1](Tables_and_Figures/Fig.%201.%20Examples%20of%20eight%20distinct%20types%20of%20original%20mango%20leaf%20images%20included%20in%20the%20dataset.png)

---

## Fig. 2: Structure of the Directory of the Dataset

![Fig2](Tables_and_Figures/Fig.%202.%20Structure%20of%20the%20directory%20of%20the%20dataset.png)

---

## Fig. 3: Flowchart Showing the Data Preparation Steps

![Fig3](Tables_and_Figures/Fig.%203.%20Flowchart%20showing%20the%20data%20preparation%20steps.png)

---

## Fig. 4: Sample of Augmented Images of MangoLeafVarietyBD Dataset (for Experimental Model Training)

![Fig4](Tables_and_Figures/Fig.%204.%20Sample%20of%20augmented%20images%20of%20MangoLeafVarietyBD%20Dataset%20%28for%20Experimental%20Model%20Training%29.png)

---

## Fig. 5: Confusion Matrices of Four Deep Learning Models on MangoLeafVarietyBD Test Set (N=1,921)

| (a) ViT-B16 | (b) DenseNet121 |
| :---------: | :-------------: |
| ![Fig5a](Tables_and_Figures/Fig.%205a.%20Confusion%20Matrix%20ViT-B16.png) | ![Fig5b](Tables_and_Figures/Fig.%205b.%20Confusion%20Matrix%20DenseNet121.png) |

| (c) GhostNetV2 | (d) ResNet18 |
| :-------------: | :-----------: |
| ![Fig5c](Tables_and_Figures/Fig.%205c.%20Confusion%20Matrix%20GhostNetV2.png) | ![Fig5d](Tables_and_Figures/Fig.%205d.%20Confusion%20Matrix%20ResNet18.png) |

---

## Fig. 6: Training Dynamics (Loss and Accuracy Curves) of Four Deep Learning Models

| (a) ViT-B16 | (b) DenseNet121 |
| :---------: | :-------------: |
| ![Fig6a](Tables_and_Figures/Fig.%206a.%20Training%20Curves%20ViT-B16.png) | ![Fig6b](Tables_and_Figures/Fig.%206b.%20Training%20Curves%20DenseNet121.png) |

| (c) GhostNetV2 | (d) ResNet18 |
| :-------------: | :-----------: |
| ![Fig6c](Tables_and_Figures/Fig.%206c.%20Training%20Curves%20GhostNetV2.png) | ![Fig6d](Tables_and_Figures/Fig.%206d.%20Training%20Curves%20ResNet18.png) |
