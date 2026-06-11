# 📋 MultiShield Consolidated Dataset Documentation

This document compiles the data cards and schemas for the four primary datasets used in training the MultiShield threat defense models.

---

## 1. Tabular Modality: UNSW-NB15
* **Task**: Anomaly & Network Intrusion Detection
* **Source**: [Kaggle - UNSW-NB15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
* **Details**:
  * **Compressed Size**: ~149 MB (`data/behavioral_anomalies.zip`)
  * **Total Samples**: 257,673 records (Pre-split: 82,332 Train / 175,341 Test)
  * **True Binary Class Distribution**: Normal: **36.1%** / Attack: **63.9%** (Majority class is Attack).
  * **Features**: 49 features including TCP/UDP states, connection duration, source/destination bytes, packet counts, protocols, and transaction logs.
  * **Known Issues**:
    * Severe imbalance across the 9 attack categories (e.g. Generic has 58,871 samples, while Worms only has 174).
    * Missing values exist in the `attack_cat` column for benign traffic (representing Normal flow).
    * **UTF-8 Byte Order Mark (BOM)** is present in the first column header (`\ufeffid` instead of `id`), which pandas strips but standard manual lookups must expect.

---

## 2. Text Modality: Phishing Emails Combined Corpus
* **Task**: Identifying Social Engineering Phishing urgencies
* **Source**: [Kaggle - Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
* **Details**:
  * **Compressed Size**: ~77 MB (`data/phishing_emails.zip`)
  * **Total Samples**: 82,486 emails
  * **Class Distribution**: Phishing: 52.0% / Legitimate: 48.0% (Nearly Balanced)
  * **Features Schema**:
    * `text_combined`: The raw email text body.
    * `label`: Binary target (1 = Phishing, 0 = Safe/Legitimate).
  * **Sub-source databases**: CEAS_08 (39,154), Enron (29,767), Ling (2,859), Nazario (1,565), Nigerian_Fraud (3,332), SpamAssasin (5,809).
  * **Known Issues**: High schema heterogeneity across sub-sources. The Enron corpus represents corporate emails, while Nigerian Fraud emails contain heavy spam indicators.

---

## 3. Text Modality: Fake and Real News Dataset
* **Task**: Narrative Disinformation classification
* **Source**: [Kaggle - Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
* **Details**:
  * **Compressed Size**: ~41 MB (`data/fake _news.zip`)
  * **Total Samples**: 44,898 articles
  * **Class Distribution**: Fake: 52.3% / Real: 47.7% (Nearly Balanced)
  * **Features Schema**: `title` (headline), `text` (body content), `subject`, and `date`.
  * **Known Issues**: Potential topic bias (e.g. real news contains strict Reuters headers which models can learn as a shortcut, which are stripped during preprocessing to ensure robustness).

---

## 4. Visual Modality: 140k Real and Fake Faces
* **Task**: Visual GAN texture identification
* **Source**: [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
* **Details**:
  * **Compressed Size**: ~3.8 GB (`data/deepfake.zip`)
  * **Total Samples**: 140,000 images (Pre-split: 100k Train / 20k Valid / 20k Test)
  * **Class Distribution**: 50.0% Real / 50.0% Fake (Perfectly Balanced)
  * **Image Specs**: 256x256 pixels, JPEG format, RGB channels.
  * **Fake Generation Source**: StyleGAN2 architectures.
  * **Known Issues**: High-frequency GAN artifacts are invisible to the naked eye. Requires convolutional filters to map color frequency anomalies.
