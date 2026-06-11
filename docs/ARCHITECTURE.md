# 🏛️ MultiShield Technical Architecture Overview

This document provides a detailed technical breakdown of the deep learning architectures, preprocessing pipelines, and enforcement endpoints configured in the MultiShield framework.

---

## 1. Tabular Modality: DeepTabularMLP
* **Task**: Anomaly/Intrusion Detection (UNSW-NB15 flow logs).
* **Architecture**:
  * Multi-layer feed-forward Multi-Layer Perceptron (MLP) built dynamically based on inputs.
  * Layer blocks: Linear -> Batch Normalization -> GELU -> Dropout.
  * Configured with 4 blocks of hidden units scaling down from `hidden_dim=128`.
  * Regularized with a Dropout probability of `0.30`.
* **Preprocessing Pipeline**:
  * **Categorical Handling**: One-hot encoding mapping unseen labels to a vector of zeros (`OneHotEncoder(handle_unknown='ignore')`).
  * **Numerical Handling**: Clips extreme values, applies a `log1p` transformation to reduce tail weight, and scales features via `RobustScaler` to limit outlier impact.
  * **Missing Values**: Dynamically constructs missing columns and fills them with default placeholders (`Unknown` for categorical, `0.0` for numerical).
  * **Imbalance Treatment**: SMOTE (Synthetic Minority Over-sampling Technique) resamples training vectors to balance attack classes.

---

## 2. Text Modality: TextLSTM
* **Task**: Semantic auditing of Fake News headlines and Phishing email contents.
* **Architecture**:
  * Word Embedding Layer (projects token indices to 128 dimensions).
  * Two-layer Bidirectional LSTM (hidden dimension `128` per direction, resulting in a concatenated sequence context of `256` dimensions).
  * Global average pooling + max pooling layers to aggregate sequence context over time.
  * Linear projection block -> Dropout (`0.3`) -> Sigmoid threat verdict.
* **Preprocessing Pipeline**:
  * Clean text sequence (HTML stripping, URL tokenization `[URL]`, lowcase, publication prefix headers stripping).
  * Integer encoding mapping tokens using terms vocabulary builder (`SimpleTokenizer` capped at 10,000 common words).
  * Pads sequences to a fixed token sequence limit (`max_length=256`).
  * Yields token index sequences alongside a tensor of true sequence lengths for efficient `pack_padded_sequence` execution in PyTorch.

---

## 3. Visual Modality: ResNet-18 Spatial Scanner
* **Task**: Exposing Generative Adversarial Network (StyleGAN2) face portrait texture blends.
* **Architecture**:
  * Base feature extractor: ResNet-18 (pre-trained ImageNet weights, frozen features representation layers).
  * Custom Classification Head: Replaces the final fully connected layer with a Linear projector mapping ImageNet activations to a single class logit.
* **Preprocessing & Test-Time Augmentation (TTA)**:
  * Resizes input JPEG/PNG images to a standard visual shape (`256 x 256` pixels).
  * Normalizes color channels with `mean=0.5, std=0.5`.
  * **TTA execution**: The API endpoint feeds the original staged image and a horizontally flipped copy through the ResNet layers. The final prediction is calculated as the average sigmoid probability of both views, utilizing a calibrated threshold of `0.4` to optimize fake GAN texture identification sensitivity.

---

## 4. Operational API & Frontend Integration

```
                                [Client browser UI]
                                         |
                                (HTTP POST Payload)
                                         v
                            [FastAPI Enforcement App]
                                         |
                        (Transforms & Tensor allocations)
                                         v
                              [PyTorch CUDA Runtime]
                                         |
                             (Sigmoid Verdict Result)
                                         v
                              [UI Gauge & Live Logs]
```

* **Startup Cache**: Models and preprocessor structures are loaded once during application initialization (`@app.on_event("startup")`) to avoid filesystem I/O overhead on live client calls.
* **Timeout Shield**: Predictions taking longer than 30 seconds are terminated via a concurrent `run_with_timeout` asynchronous wrapper, returning a gateway timeout code and preventing GPU resource starvation.
