# 13. Transformer Results (DistilBERT Token Classifier)

## Reported Evaluation Metrics

- eval_loss: 0.1050
- eval_precision: 0.9567
- eval_recall: 0.9609
- eval_f1: 0.9588
- eval_accuracy: 0.9744

## Baseline vs Transformer

- Baseline chunk-F1: 0.6688
- Transformer chunk-F1: 0.9588
- Absolute gain: 0.2900 F1

## Analysis

- Contextual embeddings provide large gains in phrase boundary decisions.
- The transformer is significantly more robust across chunk types.
- This validates current research trends favoring pretrained contextual models for sequence labeling.
