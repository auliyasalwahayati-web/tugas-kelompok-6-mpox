<div align="center">

# 🦠 Mpox Skin Lesion Classification

### Group 6 — Deep Learning Image Classification

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![CNN](https://img.shields.io/badge/Deep%20Learning-CNN-orange)
![Task](https://img.shields.io/badge/Task-Image%20Classification-purple)

**A deep learning project for classifying skin lesion images into six categories.**

</div>

---

## 📌 Overview

This project investigates deep learning models for skin lesion image classification in the context of Mpox. It compares three model configurations: **Model 1 (SimpleCNN), Model 2 (SimpleCNNReg), and EfficientNet-B0**.

The experiments use **5-fold cross-validation**. Each fold records the best validation loss, the epoch at which it was achieved, and training duration.

> **Disclaimer:** This project is intended for academic and research purposes only. It is not a medical diagnostic system and should not replace professional medical advice or examination.

## 🎯 Objectives

- Develop CNN-based models for skin lesion image classification.
- Compare model configurations using validation loss and training time.
- Evaluate training behavior across five folds.
- Identify a model configuration that balances validation performance and computational efficiency.

## 🧠 Model Configurations

| Model | Architecture | Input Size | Batch Size | Maximum Epochs | Learning Rate |
|---|---|---:|---:|---:|---:|
| Model 1 | SimpleCNN | 100 × 100 | 32 | 100 | 0.001 |
| Model 2 | SimpleCNNReg | 100 × 100 | 16 | 20 | 0.0001 |
| EfficientNet-B0 | EfficientNet-B0 (pretrained) | 224 × 224 | 16 | 20 | 0.0001 |

*Configuration details should be cross-checked against the current `config.py` before final submission.*

## ⚙️ Methodology

1. Load image data and labels using the repository data utilities.
2. Prepare images according to the input dimensions for each model.
3. Train each model using the configured optimizer and loss function.
4. Monitor validation loss and accuracy during training.
5. Save the checkpoint associated with the lowest validation loss.
6. Run experiments across **5 folds**.
7. Record best epoch, best validation loss, and elapsed training time.

## 📊 Experimental Results

### Summary — 5-Fold Cross-Validation

| Model | Mean Best Validation Loss | Mean Best Epoch | Mean Time / Fold (min) | Total Time (min) |
|---|---:|---:|---:|---:|
| Model 1 – SimpleCNN | 1.5584 | 1.4 | 3.87 | 19.33 |
| Model 2 – SimpleCNNReg | 1.4524 | 5.4 | **0.95** | 4.74 |
| EfficientNet-B0 | **0.5941** | 1.6 | 15.88 | 79.38 |

**Best mean validation loss:** EfficientNet-B0 (0.5941)  
**Fastest average training time per fold:** Model 2 (0.95 minutes)

### Model 1 — SimpleCNN

| Fold | Best Epoch | Best Validation Loss | Time (minutes) |
|---:|---:|---:|---:|
| 1 | 1 | 1.6509 | 2.70 |
| 2 | 2 | 1.3321 | 3.70 |
| 3 | 2 | 1.6671 | 4.72 |
| 4 | 1 | 1.5324 | 5.04 |
| 5 | 1 | 1.6094 | 3.18 |

### Model 2 — SimpleCNNReg

| Fold | Best Epoch | Best Validation Loss | Time (minutes) |
|---:|---:|---:|---:|
| 1 | 3 | 1.5192 | 0.84 |
| 2 | 10 | 1.1751 | 1.07 |
| 3 | 3 | 1.6466 | 0.82 |
| 4 | 6 | 1.2907 | 1.01 |
| 5 | 5 | 1.6303 | 1.01 |

### EfficientNet-B0

| Fold | Best Epoch | Best Validation Loss | Time (minutes) |
|---:|---:|---:|---:|
| 1 | 1 | 0.6160 | 13.59 |
| 2 | 3 | 0.4551 | 19.72 |
| 3 | 1 | 0.8152 | 15.55 |
| 4 | 2 | 0.4174 | 16.25 |
| 5 | 1 | 0.6666 | 14.27 |

### Average Validation Loss

![Average Best Validation Loss](mpox_mean_validation_loss.png)

### Training Time Comparison

![Average Training Time](mpox_training_time.png)

## 🔎 Discussion

Based on the recorded results, EfficientNet-B0 achieved the lowest average best validation loss (**0.5941**) among the three model configurations. This indicates that it performed better in terms of the recorded validation-loss metric during these experiments.

Model 2 had the shortest average training duration, at approximately **0.95 minutes per fold**, while EfficientNet-B0 required approximately **15.88 minutes per fold**. Model 1's average best validation loss was **1.5584**, and Model 2's was **1.4524**.

These results illustrate a trade-off between validation loss and training time. However, validation loss alone is not sufficient to determine classification performance. Accuracy, precision, recall, F1-score, and a confusion matrix should also be reported when available.

## 🔍 Error Analysis

Further analysis can be conducted using the confusion matrix and classification report to identify:

- Which class is most frequently misclassified.
- Which categories are often confused with one another.
- Per-class precision, recall, and F1-score.
- Possible effects of class imbalance or image quality.

*Add the actual confusion matrix and classification report here when available.*

## 🗂️ Repository Structure

```text
tugas-kelompok-6-mpox/
├── Models/
├── Results/
│   ├── model1/
│   ├── model2/
│   └── efficientnetb0/
├── Utils/
├── config.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```

## 🛠️ Installation

```bash
git clone https://github.com/auliyasalwahayati-web/tugas-kelompok-6-mpox.git
cd tugas-kelompok-6-mpox
pip install -r requirements.txt
```

## 🚀 Training

Run the training script with the model name configured in `config.py`:

```bash
python train.py --model model1
python train.py --model model2
python train.py --model efficientnetb0
```

A quick test can be run with fewer folds and epochs, if supported by the training script:

```bash
python train.py --model model1 --folds 1 --epochs 2
```

## 📁 Training Outputs

The repository may generate model checkpoints, per-fold training plots, and CSV history files under `Checkpoints/` and `Results/`. Check the actual output directory and file names generated by your scripts.

## 🧰 Technologies

- Python
- PyTorch
- CNN architectures
- EfficientNet-B0
- NumPy
- Matplotlib

See `requirements.txt` for the project's dependencies.

## 👥 Team Members

| No. | Name | Contribution |
|---:|---|---|
| 1 | Member 1 | — |
| 2 | Member 2 | — |
| 3 | Member 3 | — |
| 4 | Member 4 | — |

*Replace the placeholders with the actual team members and their contributions.*

## 📚 References

- [Project Repository](https://github.com/auliyasalwahayati-web/tugas-kelompok-6-mpox)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

---

<div align="center">

**Group 6 — Mpox Skin Lesion Classification**

*Academic project — not intended for clinical diagnosis.*

</div>
