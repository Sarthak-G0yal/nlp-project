# 4. Research Background

## Historical Context

Text chunking became a standard sequence-labeling benchmark with the CoNLL-2000 shared task. Early systems relied on:

- transformation-based learning
- memory-based methods
- SVM and CRF pipelines with handcrafted features

## Evolution of Methods

1. Feature-engineered linear models and CRFs
2. BiLSTM-CRF neural architectures
3. Transformer encoders with token-classification heads
4. Advanced sequence heads and span-aware modeling

## Current Standard Practice

Today, a strong practical baseline is transformer fine-tuning with BIO labels, evaluated with seqeval-style chunk metrics.
