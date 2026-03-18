# Results

This directory contains all evaluation artifacts generated from the ECG-PEFT Bench experiments.
Results are organized into three subdirectories corresponding to different output types.

---

## Directory Structure
```
results/
├── figures/               # Training curves and ROC plots
├── confusion_matrices/    # Per-model, per-method confusion matrix plots
└── metrics_tables/        # Accuracy, F1, AUC summary tables
```

---

## Metrics Reference

All models were evaluated on held-out test splits using the following metrics:

| Metric | Description | Clinical Relevance |
|---|---|---|
| **Accuracy** | Overall correct classification rate | Baseline performance signal |
| **Precision** | TP / (TP + FP) — of predicted positives, how many are real | Minimizing false alarms |
| **Recall** | TP / (TP + FN) — of actual positives, how many were caught | Critical: missing a cardiac event is costly |
| **F1 Score** | Harmonic mean of precision and recall | Balanced measure for imbalanced ECG data |
| **AUC** | Area under the ROC curve | Threshold-independent discriminative ability |
| **Confusion Matrix** | Full TP / TN / FP / FN breakdown per class | Directional error analysis |

---

## Summary of Best Results

| Model | Method | Accuracy | F1 | AUC |
|---|---|---|---|---|
| **Wav2Vec2** | **LoRA** | **0.5836** | **0.5889** | **0.6196** |
| HuBERT | Adapter | Higher Recall | — | — |
| ECG-FM | Adapter | Higher Precision | — | — |

**Key takeaways:**
- **Wav2Vec2 + LoRA** achieves the best balanced performance across accuracy, F1, and AUC
- **HuBERT** adapter variants favor higher recall — preferred in clinical settings where false negatives carry higher cost
- **ECG-FM** adapter variants favor higher precision — more conservative, fewer false alarms
- **LoRA** consistently offers the best tradeoff between parameter efficiency and downstream performance

---

## Full Analysis

Complete methodology, per-model breakdowns, ablation discussion, and all plots are documented in:

📎 [`report/ecg-peft-benchmark-report.pdf`](../report/ecg-peft-benchmark-report.pdf)