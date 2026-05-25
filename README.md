# Hinglish Hate Speech Detection using BiLSTM

> Detecting hate speech in code-mixed Hindi-English (Hinglish) social media text using deep learning — with a comparative study of RNN, LSTM, and BiLSTM architectures.

---

## Overview

This project tackles the challenging problem of hate speech detection in **Hinglish** — a code-mixed blend of Hindi and English commonly found on social media platforms like Reddit and Twitter. It benchmarks three sequential deep learning models (RNN, LSTM, BiLSTM) and demonstrates that **Bidirectional LSTM achieves the best performance** by capturing context from both directions in noisy, informal text.

---

## Problem Statement

Hate speech detection in Hinglish is uniquely difficult due to:

- **Code-mixing** — fluid switching between Hindi and English within a sentence
- **Informal language** — slang, abbreviations, and non-standard spelling
- **Implicit expressions** — sarcasm, irony, and context-dependent hostility
- **Long-range dependencies** — meaning that spans multiple tokens or clauses
- **Limitations of classical ML** — bag-of-words models fail to capture sequential semantics

---

## Objectives

- Build a hate speech classifier specifically for Hinglish text
- Compare RNN, LSTM, and BiLSTM on the same dataset and preprocessing pipeline
- Apply robust preprocessing tailored to noisy social media data
- Evaluate using accuracy, precision, recall, and macro F1-score

---
## Results

| Model   | Accuracy | Precision | Recall | Macro F1 |
|---------|----------|-----------|--------|----------|
| RNN     | 85.20%   | 0.84      | 0.82   | 0.83     |
| LSTM    | 91.40%   | 0.91      | 0.89   | 0.90     |
| **BiLSTM** | **94.86%** | **0.95** | **0.93** | **0.942** |

**BiLSTM outperforms both RNN and LSTM** across all metrics, with notable improvements in detecting sarcasm and slang-based expressions where bidirectional context is critical.

---

## Model Architecture

### RNN
A basic sequential model that processes text left-to-right. Suffers from vanishing gradients on longer sequences.

### LSTM
Adds gating mechanisms (input, forget, output gates) to retain long-range dependencies more reliably than RNN.

### BiLSTM *(Best Performer)*
Processes the sequence in **both forward and backward directions**, producing richer contextual representations at each timestep. Architecture:
```
Input → Embedding Layer → Stacked BiLSTM Layers → Dense Layer → Sigmoid Output
```
---
## Pipeline

```
Data Collection → Preprocessing → Tokenization → Model Training → Evaluation → Analysis
```

**Preprocessing steps:**
- Lowercase conversion
- URL and mention removal
- Special character and number removal
- Text normalization and slang handling
- Sequence tokenization and padding

---

## Dataset

Code-mixed Hinglish text sourced from publicly available datasets (Kaggle and Reddit-style discussions). The merged dataset contains:

- Hindi-English mixed sentences in informal writing
- Labeled samples for hate and non-hate speech
- Sarcastic and context-dependent expressions

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Evaluation | Scikit-learn |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Binary Cross-Entropy |
| Optimizer | Adam |
| Output Activation | Sigmoid |
| Metrics | Accuracy, Precision, Recall, Macro F1 |

---

## Future Work

- [ ] Fine-tune transformer-based models (BERT, MuRIL, RoBERTa) on Hinglish
- [ ] Expand and diversify the training dataset
- [ ] Improve sarcasm and implicit hate detection
- [ ] Build a real-time detection API
- [ ] Deploy as a web application

---

## Project Structure

```
├── data/                  # Raw and preprocessed datasets
├── notebooks/             # Jupyter notebooks for exploration and training
├── models/                # Saved model weights
├── src/
│   ├── preprocess.py      # Text cleaning and tokenization
│   ├── model.py           # RNN, LSTM, BiLSTM definitions
│   └── evaluate.py        # Metrics and result visualization
├── results/               # Training curves and evaluation outputs
└── README.md
```

---

## License

This project is for academic and research purposes.
