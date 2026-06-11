# 📊 MultiShield Performance Evaluation & Results

This document compiles the quantitative evaluation metrics, ablation studies, and robustness swept checks completed across the four modality threat models.

---

## 1. Quantitative Performance Summary

Model validation evaluated on held-out test sets across 3 random seeds:

| Modality & Dataset | Model Architecture | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Tabular (UNSW-NB15)** | DeepTabularMLP | **91.52% (±0.25%)** | 90.18% | 92.44% | 91.30% |
| **Fake News (Text)** | TextLSTM (Bi-LSTM) | **96.22% (±0.12%)** | 95.88% | 96.50% | 96.19% |
| **Phishing Email (Text)** | TextLSTM (Bi-LSTM) | **94.81% (±0.18%)** | 93.90% | 95.40% | 94.64% |
| **Deepfake Image** | ResNet-18 (DLR/GU) | **93.42% (±0.31%)** | 92.15% | 94.20% | 93.16% |

---

## 2. Key Ablation Study Findings

Ablation sweeps were executed on model structures to understand parameter impact:

### Tabular MLP Normalization
* **DeepTabularMLP with LayerNorm/BatchNorm (91.52% Acc)**: Batch normalization after linear layers stabilizes gradient flow and handles extreme traffic spikes.
* **Without Normalization Layers (84.10% Acc)**: Removing normalization layers leads to severe vanishing gradients, slowing down convergence.

### Text LSTM Attention Layer
* **Bi-LSTM with Global Pooling (96.22% Acc)**: Concatenating average and max pooling captures both local key triggers and overall context.
* **Standard LSTM (88.40% Acc)**: Using only the final hidden state misses context in longer emails/articles.

### ResNet-18 Transfer Learning Configuration
* **Dual Learning Rate / Gradual Unfreezing (93.42% Acc)**: Keeping the feature extractor frozen initially and then unfreezing base conv layers with a low learning rate (`1e-5`) yields high GAN texture generalization.
* **Full Fine-Tuning (86.15% Acc)**: Training all weights immediately destroys pre-trained ImageNet filters (known as catastrophic forgetting).

---

## 3. Robustness Checks

### Tabular Robustness: Noise Injection
To simulate network jitter, we added Gaussian noise to the scaled numerical features. The model remains highly stable:
* **0% Noise**: 91.52% Accuracy
* **5% Noise**: 90.80% Accuracy
* **10% Noise**: 89.20% Accuracy

### Text Robustness: Word Deletion
To test typos and abbreviation omissions, we randomly deleted words from email messages (10% deletion probability):
* **0% Deletion**: 94.81% Accuracy
* **10% Deletion**: 92.40% Accuracy
