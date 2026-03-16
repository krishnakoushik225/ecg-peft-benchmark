# ECG-PEFT Bench 🫀
### LoRA vs Adapters for ECG Segment Classification · Wav2Vec2 / HuBERT / ECG-FM

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20Adapters-green)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **ECG-PEFT Bench** is a medical AI research project benchmarking Parameter-Efficient Fine-Tuning (PEFT) strategies for binary ECG segment classification across three cardiac foundation models — Wav2Vec2, HuBERT, and ECG-FM — comparing LoRA vs full adapter methods with automated evaluation harnesses.

---

## 🧬 Research Question

> *Can PEFT methods (LoRA, adapters) efficiently fine-tune cardiac foundation models for ECG segment classification while maintaining competitive performance with significantly fewer trainable parameters?*

---

## 📊 Results Summary

| Model | Method | Accuracy | F1 | AUC |
|-------|--------|----------|----|-----|
| **Wav2Vec2** | **LoRA** | **0.5836** | **0.5889** | **0.6196** |
| HuBERT | Adapter | Higher Recall | — | — |
| ECG-FM | Adapter | Higher Precision | — | — |

**Key findings:**
- **Wav2Vec2 + LoRA** achieves the best overall balance of accuracy, F1, and AUC
- **HuBERT** variants favor higher recall — useful in clinical settings where missing positives is costly
- **ECG-FM** shows stronger precision — more conservative, fewer false positives
- **LoRA consistently** provides the best tradeoff between parameter efficiency and performance

Full metrics tables, confusion matrices, and plots are in `results/` and `report/Report_Lab_3.pdf`.

---

## 🏗️ Architecture

```
ECG Segment (raw 1D waveform)
        │
        ▼
┌───────────────────────┐
│   Preprocessing       │  Normalization + windowing
└──────────┬────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│         Pretrained Foundation Model       │
│   Wav2Vec2  /  HuBERT  /  ECG-FM        │
│                                          │
│   ┌──────────────────────────────────┐   │
│   │      PEFT Layer Injection        │   │
│   │  LoRA:    W = W₀ + BA (rank r)  │   │
│   │  Adapter: bottleneck modules     │   │
│   └──────────────────────────────────┘   │
└─────────────────────┬────────────────────┘
                      │
                      ▼
           Binary Classification Head
           (ECG segment: normal / abnormal)
                      │
                      ▼
          ┌───────────────────────┐
          │    Evaluation Suite   │
          │  Accuracy, Precision  │
          │  Recall, F1, AUC      │
          │  Confusion Matrices   │
          └───────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA-capable GPU recommended (CPU works for inference)

### Installation

```bash
git clone https://github.com/krishnakoushik225/ecg-peft
cd ecg-peft
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data Setup

Place your ECG dataset under:
```
data/
```
Ensure the notebook paths reference the correct location.

### Run the Notebook

```bash
jupyter notebook notebooks/ecg_peft_benchmark.ipynb
```

Open and execute `notebooks/ecg_peft_benchmark.ipynb` end-to-end — covers preprocessing, PEFT integration, training, and full evaluation.

---

## 🔬 Evaluation Suite

Each model/method combination is evaluated on held-out test splits with:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct classification rate |
| **Precision** | True positive rate among predicted positives |
| **Recall** | True positive rate among actual positives (critical for clinical use) |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC** | Area under the ROC curve |
| **Confusion Matrix** | Full breakdown of TP, TN, FP, FN per class |

Results are stored in `results/metrics_tables/`, plots in `results/figures/`, and confusion matrices in `results/confusion_matrices/`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Foundation Models | Wav2Vec2, HuBERT, ECG-FM |
| PEFT Methods | LoRA (Low-Rank Adaptation), Adapter modules |
| ML Framework | PyTorch + HuggingFace Transformers |
| PEFT Library | HuggingFace PEFT |
| Notebook | Jupyter |
| Evaluation | scikit-learn (accuracy, F1, AUC, confusion matrix) |

---

## 📁 Project Structure

```
ecg-peft-benchmark/
├── notebooks/
│   └── ecg_peft_benchmark.ipynb   # End-to-end pipeline
├── report/
│   └── Report_Lab_3.pdf           # Full technical report + analysis
├── results/
│   ├── figures/                   # Training curves, ROC plots
│   ├── confusion_matrices/        # Per-model confusion matrices
│   └── metrics_tables/            # Accuracy, F1, AUC tables
└── README.md
```
---

## 📄 Full Report

Complete technical analysis, methodology, and evaluation artifacts:

📎 [`report/Report_Lab_3.pdf`](report/Report_Lab_3.pdf)

---

## 🔭 Roadmap

- [ ] QLoRA experiments (4-bit quantization)
- [ ] Cross-dataset generalization (PTB-XL, PhysioNet)
- [ ] MLflow + W&B experiment tracking integration
- [ ] Multi-class arrhythmia classification (beyond binary)
- [ ] ONNX export for clinical deployment packaging
- [ ] Model merging experiments (DARE, TIES)

---

## 📄 License

MIT — free to use and build on. If you use this in research, a citation would be appreciated.

---

*Built by [Krishna Koushik Unnam](https://github.com/krishnakoushik225) · M.S. Computer Science, University of South Florida*
