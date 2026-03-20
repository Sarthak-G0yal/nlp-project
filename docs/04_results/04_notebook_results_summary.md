# Combined Results Summary (Main + Baselines Notebooks)

## Scope

This summary combines outcomes from:
- Notebook.ipynb (baseline logistic regression vs transformer DistilBERT)
- Other_Shallow_Parsing_Baselines.ipynb (MultinomialNB, LogisticRegression, BiLSTM, optional CRF)

## 1) Main Notebook Results

### Baseline vs Transformer (core project comparison)

- Baseline chunk_f1: 0.6688
- Transformer chunk_f1: 0.9586
- Absolute chunk_f1 gain: 0.2898
- Relative chunk_f1 gain: 43.33%
- Chunk_f1 error reduction: 87.49%

Additional strong improvements:
- Chunk accuracy: 0.7457 -> 0.9743
- Token macro F1: 0.2678 -> 0.6917

Interpretation:
- Contextual transformer modeling dramatically improves chunk boundary quality over sparse local features.
- The largest practical benefit is the error reduction at chunk level, which is highly relevant for downstream information extraction.

## 2) Other Baselines Notebook Results

### Chunk-level comparison

- CRF (optional): precision 0.931, recall 0.930, f1 0.931, accuracy 0.955
- LogisticRegression: precision 0.885, recall 0.910, f1 0.898, accuracy 0.939
- MultinomialNB: precision 0.850, recall 0.897, f1 0.873, accuracy 0.917
- BiLSTM: precision 0.787, recall 0.818, f1 0.803, accuracy 0.888

### Token-level behavior (from extended comparison visuals)

- LogisticRegression leads among non-CRF baselines on token metrics.
- MultinomialNB is competitive but below LogisticRegression.
- BiLSTM under current setup (3000-train subset, current epochs/hyperparameters) is below sparse-feature classical models.

Interpretation:
- Strong feature-based classical methods remain robust baselines for CoNLL-2000.
- CRF is the best method among the non-transformer baseline set in this run.
- Neural baseline quality is sensitive to training size and hyperparameter tuning.

## 3) Cross-Notebook Takeaways

1. Best overall in this project: Transformer (Notebook.ipynb).
2. Best non-transformer baseline: CRF (optional) in Other_Shallow_Parsing_Baselines.ipynb.
3. Best always-available lightweight baseline: LogisticRegression.
4. Recommendation for reporting:
   - Use transformer as primary model.
   - Use LogisticRegression and CRF as baseline references.
   - Mention that BiLSTM can improve with larger training data and tuning.

## 4) Suggested Reporting Table

Include this order in your final report:
- MultinomialNB
- LogisticRegression
- BiLSTM
- CRF (optional)
- DistilBERT Transformer

This presents a clear progression from simple probabilistic to structured classical to neural transformer methods.
