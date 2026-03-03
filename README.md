# ECG-PEFT Bench  
### LoRA vs Adapters for ECG Segment Classification (Wav2Vec2 / HuBERT / ECG-FM)

---

## Overview

This project benchmarks **Parameter-Efficient Fine-Tuning (PEFT)** strategies for **binary ECG segment classification** using three pretrained backbones:

- **Wav2Vec 2.0**
- **HuBERT**
- **ECG-FM** (ECG foundation model)

Two PEFT methods are compared across all backbones:

- **Adapters**
- **LoRA (Low-Rank Adaptation)**

The pipeline covers preprocessing, PEFT integration, training, and evaluation with:
**Accuracy, Precision, Recall, F1, AUC**, plus confusion matrices and plots.

---

## Key Highlights

- End-to-end notebook pipeline for PEFT on 1D biomedical signals  
- Consistent metric reporting across validation and test splits  
- Comparative analysis of LoRA vs adapters for each backbone  
- Report includes detailed metrics tables, confusion matrices, and qualitative insights

---

## Results Summary (from report)

Best overall test performance:

- **Wav2Vec2 + LoRA**: Accuracy **0.5836**, F1 **0.5889**, AUC **0.6196**

Additional observations:

- **HuBERT** variants tend to favor higher recall depending on configuration.
- **ECG-FM** can be more conservative, showing stronger precision in some settings.
- LoRA generally provides a strong balance between parameter efficiency and performance.

For full metrics, plots, and confusion matrices, see the report.

---

## Repository Structure

```
ecg-peft-benchmark/
│
├── notebooks/
│   └── ecg_peft_benchmark.ipynb
│
├── report/
│   └── Report_Lab_3.pdf
│
├── results/
│   ├── figures/
│   ├── confusion_matrices/
│   └── metrics_tables/
│
└── README.md
```

---

## Getting Started

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the notebook

Open and execute:

- `notebooks/ecg_peft_benchmark.ipynb`

---

## Data Notes

If the dataset is not included in this repository, place it under:

- `data/`  

and ensure the notebook paths reference the correct location.

---

## Checkpoints / Large Files

Do not commit large model checkpoints (`.pt`, `.pth`, `.ckpt`) directly to GitHub due to the 100MB per-file limit.

Recommended approach:

- Keep checkpoints local, or  
- Upload via GitHub Releases / Google Drive / Hugging Face and link them here.

---

## Report

Full technical report and evaluation artifacts are available at:

- `report/Report_Lab_3.pdf`

---

## Author

Krishna Koushik Unnam  
M.S. Computer Science & Engineering  
University of South Florida

---