# Combined Results Summary (Main + Baselines + Scaling + Bridge)

## Scope

This summary combines outcomes from:
- Notebook.ipynb (baseline logistic regression vs transformer DistilBERT)
- Other_Shallow_Parsing_Baselines.ipynb (MultinomialNB, LogisticRegression, BiLSTM, optional CRF)
- ipynb/Domain_Specific_Tokenization_Chunk_Aware_Preprocessing.ipynb (domain normalization, alignment-aware scoring, retraining/sweep gate)
- ipynb/Larger_Transformer_Comparison.ipynb (scaling analysis with expanded model pool)
- ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb (event-bridge + confidence/abstention + cost-aware routing)

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

## 3) Transformer vs Other Baselines (Direct Comparison)

Using DistilBERT chunk_f1 = 0.9586 from the main notebook as reference:

| Model | chunk_f1 | Gap vs Transformer | Relative error reduction by Transformer |
|---|---:|---:|---:|
| DistilBERT Transformer | 0.9586 | 0.0000 | - |
| CRF (optional) | 0.9307 | 0.0279 | 40.3% |
| LogisticRegression | 0.8975 | 0.0611 | 59.6% |
| MultinomialNB | 0.8726 | 0.0860 | 67.5% |
| BiLSTM | 0.8025 | 0.1561 | 79.0% |

Notes:
- CRF remains a strong non-transformer baseline, but transformer still leads by +0.0279 chunk_f1.
- LogisticRegression remains the best always-available lightweight baseline.
- BiLSTM under this configuration (subset + current hyperparameters) is clearly below sparse-feature classical baselines.

## 4) Cross-Notebook Takeaways

1. Best overall in this project: Transformer (Notebook.ipynb).
2. Best non-transformer baseline: CRF (optional) in Other_Shallow_Parsing_Baselines.ipynb.
3. Best always-available lightweight baseline: LogisticRegression.
4. Recommendation for reporting:
   - Use transformer as primary model.
   - Use LogisticRegression and CRF as baseline references.
   - Mention that BiLSTM can improve with larger training data and tuning.

## 5) Suggested Reporting Table

Include this order in your final report:
- MultinomialNB
- LogisticRegression
- BiLSTM
- CRF (optional)
- DistilBERT Transformer

This presents a clear progression from simple probabilistic to structured classical to neural transformer methods.

## 6) Scaling Notebook Update (New Models)

### Newly added models in the scaling notebook

The model pool in `ipynb/Larger_Transformer_Comparison.ipynb` now includes:

- distilbert/distilbert-base-uncased
- bert-base-uncased
- roberta-base
- google/bert_uncased_L-2_H-128_A-2 (NanoBERT-like compact variant)
- prajjwal1/bert-tiny
- huawei-noah/TinyBERT_General_4L_312D

### Current measured vs failed state

- Measured in latest scaling execution: DistilBERT, BERT-base, NanoBERT-like, TinyBERT
- Failed in latest scaling execution:
   - roberta-base (CUDA OOM)
   - prajjwal1/bert-tiny (tokenizer backend dependency: sentencepiece/tiktoken)

Current recommendation from the notebook decision cell:

- DistilBERT remains the baseline.
- Bigger models are not clearly worth it under the current threshold in this run.
- BERT-base improves F1 but only by +0.0039, below the +0.005 decision cutoff.

Interpretation:

- The scaling study now includes actual low-cost frontier points (NanoBERT-like and TinyBERT).
- Tiny models trade large quality loss for speed/size gains.
- A clean final comparison still requires successful RoBERTa and bert-tiny reruns in a compatible environment.

## 7) Chunk-to-Event Bridge Results

### What was added

- explicit event schema and role requirements
- confidence + abstention mechanism for event extraction
- cost-aware model recommendation layer
- pending-model-safe recommendation tables (for profiles that are not yet synchronized in the bridge snapshot)

### Current executed outputs

- measured model profile includes distilbert, bert-base-uncased, roberta-base
- bridge notebook currently still shows tiny-model rows as pending (it has not yet been refreshed with the newest scaling execution)
- recommendation output (current thresholds): roberta-base in both quality-first and latency-first views
- email pipeline run summary: accepted events 5/5, total cost estimate 38.00, cost per accepted event 7.60

Interpretation:

- The project now has a direct bridge from chunk predictions to actionable event records.
- The analysis layer can compare extraction quality and compute constraints together.
- Tiny-model integration is structurally complete; scaling measurements are now available and can be propagated by rerunning bridge/profile cells.

## 8) Domain-Specific Tokenization Innovation Notebook Update

Notebook: `ipynb/Domain_Specific_Tokenization_Chunk_Aware_Preprocessing.ipynb`

### What was added

- deterministic domain normalization rules for emails/URLs/currency/time/date/ID patterns,
- anti-regression assertions for known normalization failure modes,
- alignment-aware label scoring replacing strict token-length evaluability,
- tokenizer-aware retraining plus iterative sweep with deployment-threshold gating.

### Key measured outcomes

- alignment-aware evaluable coverage is high (~0.95-0.98),
- best preprocessing-only result remains tied with baseline (chunk F1 ~0.9562),
- single retrain run underperformed baseline,
- best sweep run improved to ~0.9582 (+0.0020 vs baseline) but stayed below deployment threshold (+0.005).

### Current recommendation

- keep baseline as deployment model,
- keep innovation notebook workflow as the experimentation and decision-governance path,
- continue larger sweeps/model-scale trials only if +0.005 gain threshold remains mandatory.
