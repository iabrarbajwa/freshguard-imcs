# FreshGuard — Fruit Quality Inspection

Deep learning system for automated fruit quality control. Classifies six fruit types (Apple, Banana, Guava, Lime, Orange, Pomegranate) into **Good** or **Bad** quality using transfer learning, with a Streamlit dashboard for live image inspection.

IMCS course project — **Abrar Mustafa**.

## Overview

FreshGuard addresses quality inspection in food supply chains by training image classifiers on labeled fruit photographs. Two architectures are compared: **ResNet-18** (lightweight CNN) and **ViT-Base/16** (Vision Transformer). A group-aware train/validation/test split prevents data leakage from repeated fruit samples. The Streamlit app lets users upload images and receive accept/reject decisions with confidence scores.

## Features

- **12-class fine-grained classification** (6 fruits × 2 quality levels)
- **Binary accept/reject monitoring** derived from Good vs. Bad predictions
- **Two model backends:** ResNet-18 and ViT-Base/16 (via `timm`)
- **Streamlit dashboard** with image upload, top-k predictions, and metrics view
- **Training notebook** (`FreshGuard_IMCS.ipynb`) with full Colab-compatible pipeline
- **Group split CSV** for reproducible evaluation
- **Sample test images** for quick demo without the full dataset

## Technologies

| Category | Stack |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, torchvision, timm |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Training | Jupyter Notebook |
| Image Processing | Pillow |

## Project Structure

```text
FreshGuard_IMCS/
├── app.py                              # Streamlit dashboard
├── FreshGuard_IMCS.ipynb               # Full training & evaluation notebook
├── requirements.txt                    # Python dependencies
├── results/                            # Saved evaluation metrics (CSV)
│   ├── model_comparison.csv
│   └── binary_monitoring_metrics.csv
├── Test Images/                        # Sample demo images
├── checkpoints/                        # Trained PyTorch weights
├── splits/                             # Dataset split CSV
└── freshGuardvenv/                     # Local venv (excluded from repo)
```

## Installation

### Prerequisites

- Python 3.10 or newer
- Git
- ~4 GB disk space for checkpoints (stored locally after training)

### Setup

```bash
git clone https://github.com/iabrarbajwa/freshguard-imcs.git
cd freshguard-imcs

python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Model Checkpoints

Checkpoints are included in `checkpoints/` (large files stored via Git LFS):

```text
checkpoints/
├── resnet18_clean_best.pt
└── vit_base_clean_best.pt
```

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

1. Confirm the **Project folder path** in the sidebar points to this repository root.
2. Select **ResNet-18** or **ViT-Base/16**.
3. Upload a fruit image (or use files from `Test Images/`).
4. View accept/reject decision, fruit type, quality label, confidence, and top predictions.

### Train Models

Open `FreshGuard_IMCS.ipynb` in Jupyter or Google Colab. The notebook covers:

- Data loading and group-aware splitting
- ResNet-18 and ViT-Base/16 fine-tuning
- Evaluation metrics and model comparison
- Checkpoint export to `checkpoints/`

## Results

### Multi-Class Test Performance

| Model | Parameters (M) | Test Accuracy | Macro F1 |
|---|---:|---:|---:|
| ResNet-18 | 11.18 | 99.35% | 0.993 |
| ViT-Base/16 | 85.81 | 98.34% | 0.980 |

### Binary Accept/Reject Monitoring

| Model | Binary Accuracy | False Accept | False Reject |
|---|---:|---:|---:|
| ResNet-18 | 99.82% | 0.37% | 0.00% |
| ViT-Base/16 | 99.17% | 1.10% | 0.56% |

ResNet-18 achieves the best balance of accuracy, speed, and model size for deployment. ViT-Base/16 offers competitive accuracy at higher parameter count.

## Future Improvements

- Add **Grad-CAM** and **LIME** explainability in the dashboard
- Export models to ONNX / TorchScript for production inference
- Real-time camera feed support in Streamlit
- Data augmentation pipeline for rare defect classes
- Publish checkpoints via GitHub Releases
- REST API wrapper for integration with sorting-line hardware
- Expand fruit catalog and multi-defect grading (bruise, mold, ripeness)

## Author

Abrar Mustafa — IMCS FreshGuard project.
