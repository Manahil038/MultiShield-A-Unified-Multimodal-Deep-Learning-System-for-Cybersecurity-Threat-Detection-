# MultiShield: Unified Multimodal Deep Learning System for Cybersecurity Threat Detection

A modular deep learning framework that detects cybersecurity threats across multiple data modalities — text (phishing emails, fake news), tabular (network intrusion), and images (deepfakes).

## Project Structure

```
project-root/
├── data/                # raw + processed data (gitignored)
├── notebooks/           # exploration only
│   └── 01_EDA.ipynb
├── src/
│   ├── data/            # loading, splitting
│   ├── preprocessing/   # transforms, augmentation
│   ├── models/          # model definitions
│   ├── training/        # train/eval loops
│   └── utils/           # logging, seeds, helpers
├── experiments/         # configs, run logs
├── reports/             # figures, results, progress reports
├── tests/               # smoke tests
├── requirements.txt     # pinned versions
├── README.md
└── .gitignore
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Manahil038/MultiShield-A-Unified-Multimodal-Deep-Learning-System-for-Cybersecurity-Threat-Detection-.git
cd MultiShield-A-Unified-Multimodal-Deep-Learning-System-for-Cybersecurity-Threat-Detection-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
# Place dataset zips into the data/ directory.
# See reports/ for dataset provenance details.

# 4. Reproduce the Baseline MLP Experiment
python experiments/run_baseline.py

# 5. Run smoke tests
pytest tests/ -v

# 6. Launch TensorBoard (to view Baseline tracking logs)
tensorboard --logdir experiments/
```

## Datasets

| Dataset | Modality | Task |
|---------|----------|------|
| Phishing Emails | Text | Binary Classification |
| Fake & Real News | Text | Binary Classification |
| UNSW-NB15 | Tabular | Multi-class Classification |
| FaceForensics++ | Image | Binary Classification (Pending) |

## Key Design Decisions

- **Activation:** GELU (prevents dying ReLU, aligns with Transformer baselines)
- **Weight Init:** He/Kaiming Normal (mathematically prevents vanishing gradients with GELU)
- **Loss:** BCEWithLogitsLoss (binary) / CrossEntropyLoss (multi-class)
- **Optimizer:** AdamW with ReduceLROnPlateau scheduling
- **Experiment Tracking:** TensorBoard

## Team

Team MultiShield — Deep Learning Semester Project
