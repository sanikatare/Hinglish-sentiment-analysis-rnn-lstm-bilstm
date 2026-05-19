# A Deep Learning Approach for Code-Mixed (Hinglish) Hate Speech Detection Using BiLSTM

## Overview
This project focuses on detecting hate speech in code-mixed Hinglish text using deep learning models. It implements and compares three neural network architectures: Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM (BiLSTM). The goal is to improve classification performance on informal social media text such as Reddit-style conversations, where language is noisy, slang-heavy, and context-dependent.

The study highlights the effectiveness of BiLSTM in capturing bidirectional contextual dependencies in code-mixed text compared to traditional sequential models.

---

## Problem Statement
Hate speech detection in Hinglish text is challenging due to:

- Code-mixed Hindi-English language structure
- Informal writing style and spelling variations
- Presence of slang, sarcasm, and implicit expressions
- Long-range contextual dependencies in conversations
- Limitations of traditional machine learning models in understanding sequential context

---

## Objectives
- Build a system for detecting hate speech in Hinglish text
- Compare performance of RNN, LSTM, and BiLSTM models
- Apply preprocessing techniques for noisy social media data
- Handle slang, normalization, and tokenization effectively
- Evaluate models using standard classification metrics

---

## Dataset
The dataset consists of Hinglish code-mixed text collected from publicly available sources such as Kaggle and Reddit-style social media discussions.

The dataset includes:
- Hindi-English mixed sentences
- Informal and slang-based expressions
- Sarcasm and context-dependent language
- Labeled samples for hate and non-hate speech

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Methodology

### Data Collection
Data was collected from publicly available datasets containing Hinglish and code-mixed social media text. Merged several datasets for training the model.

### Data Preprocessing
The following preprocessing steps were applied:

- Conversion to lowercase
- Removal of URLs and mentions
- Removal of special characters and numbers
- Text normalization
- Slang handling and cleaning
- Tokenization of text into sequences

---

## Model Architecture

### Recurrent Neural Network (RNN)
A basic sequential model used to capture dependencies in text but limited by vanishing gradient issues.

### Long Short-Term Memory (LSTM)
An improved sequential model that uses gating mechanisms to capture long-term dependencies.

### Bidirectional LSTM (BiLSTM)
An advanced model that processes input in both forward and backward directions, enabling better contextual understanding.

BiLSTM architecture includes:
- Embedding Layer
- Stacked BiLSTM Layers
- Dense Layer
- Sigmoid Output Layer

---

## Training Details
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam
- Evaluation Metrics: Accuracy, Precision, Recall, Macro F1-Score
- Training Strategy: Sequential training across datasets

---

## Evaluation Metrics
The models were evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-Score

---

## Results

| Model   | Accuracy | Precision | Recall | Macro F1-Score |
|---------|----------|-----------|--------|----------------|
| RNN     | 85.20%   | 0.84      | 0.82   | 0.83           |
| LSTM    | 91.40%   | 0.91      | 0.89   | 0.90           |
| BiLSTM  | 94.86%   | 0.95      | 0.93   | 0.942          |

### Key Observations
- BiLSTM achieved the best performance among all models
- It effectively captures bidirectional context in Hinglish text
- It performs better in detecting sarcasm and slang-based expressions
- It reduces misclassification compared to RNN and LSTM

---

## System Workflow
1. Data Collection
2. Data Preprocessing
3. Tokenization
4. Model Building
5. Model Training
6. Model Evaluation
7. Comparative Analysis
8. Result Visualization

---

## Future Improvements
- Implementation of transformer-based models such as BERT or RoBERTa
- Expansion of dataset size for better generalization
- Real-time hate speech detection system for social media
- Improved sarcasm and contextual understanding models
- Deployment as a web application or API

---

## Conclusion
This project demonstrates that Bidirectional LSTM outperforms RNN and LSTM models in detecting hate speech in code-mixed Hinglish text. The ability of BiLSTM to capture contextual information from both directions significantly improves classification accuracy in noisy and informal social media environments.

---

