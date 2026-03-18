# Architecture

This document describes the end-to-end pipeline of the ECG-PEFT Bench system —
from raw ECG waveform input through foundation model fine-tuning to final binary classification and evaluation.

---

## Pipeline Overview
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

## Stage Breakdown

### 1. Preprocessing
Raw ECG segments undergo normalization and windowing before being passed to the foundation model.
This ensures consistent input representation across the three model architectures.

### 2. Foundation Model (Frozen Backbone)
Three cardiac foundation models serve as the frozen encoder backbone:

| Model | Architecture | Pretraining Domain |
|---|---|---|
| **Wav2Vec2** | CNN + Transformer | Speech (adapted to ECG) |
| **HuBERT** | CNN + Transformer | Self-supervised audio |
| **ECG-FM** | Transformer | ECG-native pretraining |

The backbone weights are frozen during fine-tuning. Only PEFT modules are updated.

### 3. PEFT Layer Injection
Two parameter-efficient fine-tuning strategies are injected into the frozen backbone:

**LoRA (Low-Rank Adaptation)**
- Decomposes weight updates as `W = W₀ + BA` where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, rank `r ≪ d`
- Trains only the low-rank matrices — backbone weights remain unchanged
- Typical trainable parameter reduction: ~99% fewer parameters than full fine-tuning

**Adapter Modules**
- Inserts small bottleneck feed-forward layers between transformer blocks
- Each adapter: down-projection → non-linearity → up-projection with residual connection
- Allows task-specific adaptation while preserving pretrained representations

### 4. Binary Classification Head
A lightweight linear classification head maps the encoder's pooled output to a binary label:
- `0` — Normal ECG segment
- `1` — Abnormal ECG segment

### 5. Evaluation Suite
Each model/method combination is evaluated on a held-out test split. Outputs include:
- Per-class confusion matrices (`results/confusion_matrices/`)
- ROC curves and training loss plots (`results/figures/`)
- Aggregated accuracy, F1, AUC tables (`results/metrics_tables/`)

---

## Design Rationale

**Why PEFT over full fine-tuning?**
Cardiac foundation models contain tens of millions of pretrained parameters. Full fine-tuning on
limited ECG classification datasets risks catastrophic forgetting and overfitting. PEFT methods
preserve the pretrained representations while enabling task-specific adaptation with a fraction
of the trainable parameters — critical for medical AI where labeled data is scarce.

**Why three foundation models?**
Each model brings a different inductive bias. Comparing across Wav2Vec2, HuBERT, and ECG-FM
isolates the effect of pretraining domain (speech-adapted vs ECG-native) on downstream
classification performance — directly answering the research question.

---

## Full Technical Report

📎 [`report/ecg-peft-benchmark-report.pdf`](../report/ecg-peft-benchmark-report.pdf)