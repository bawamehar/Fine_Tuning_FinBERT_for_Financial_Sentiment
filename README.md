# Financial Sentiment Analysis using FinBERT

Fine-tuned FinBERT on financial news sentences to classify sentiment as 
positive, negative, or neutral. Built an end-to-end pipeline from data 
loading to inference using HuggingFace Transformers and PyTorch.

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 96.5% |
| Test F1 Score | 96.5% |
| Negative Precision | 100% |

---

## Dataset

**Financial PhraseBank** (sentences_75agree)
- 3,453 financial news sentences
- Labeled by 16 domain experts with 75%+ annotator agreement
- 3 classes: Negative, Neutral, Positive
- Split: 70% train / 15% validation / 15% test (stratified)

---

## Model

**ProsusAI/FinBERT** — a BERT model pre-trained on financial text, 
fine-tuned for 3-class sentiment classification.

---

## Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- Scikit-learn
- Google Colab (T4 GPU)

---

## Key Concepts Demonstrated

- Transformer fine-tuning on domain-specific data
- Stratified train/validation/test splitting for class imbalance
- Tokenization with padding, truncation and attention masking
- GPU-accelerated training with HuggingFace Trainer API
- Per-class evaluation using precision, recall and F1

---
