# 13. Transformer Results (DistilBERT Token Classifier)

## Reported Evaluation Metrics

- eval_loss: 0.1050
- eval_precision: 0.9567
- eval_recall: 0.9609
- eval_f1: 0.9588
- eval_accuracy: 0.9744

## Baseline vs Transformer

- Baseline chunk-F1: 0.6688
- Transformer chunk-F1: 0.9586
- Absolute gain: 0.2898 F1

## Transformer vs Other Baselines

Direct chunk-level comparison against models reported in `ipynb/Other_Shallow_Parsing_Baselines.ipynb`:

- Transformer (DistilBERT): chunk_f1 0.9586
- CRF (optional): chunk_f1 0.9307 (gap: 0.0279)
- LogisticRegression: chunk_f1 0.8975 (gap: 0.0611)
- MultinomialNB: chunk_f1 0.8726 (gap: 0.0860)
- BiLSTM: chunk_f1 0.8025 (gap: 0.1561)

Relative error reduction from transformer over each baseline:

- vs CRF: 40.3%
- vs LogisticRegression: 59.6%
- vs MultinomialNB: 67.5%
- vs BiLSTM: 79.0%

## Analysis

- Contextual embeddings provide large gains in phrase boundary decisions.
- The transformer is significantly more robust across chunk types.
- This validates current research trends favoring pretrained contextual models for sequence labeling.
