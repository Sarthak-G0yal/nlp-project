# Domain-Specific Tokenization Innovation Results

Notebook: ipynb/Domain_Specific_Tokenization_Chunk_Aware_Preprocessing.ipynb

## Objective

This innovation track tested whether domain-aware preprocessing can improve chunking and downstream extraction quality without immediately requiring full model replacement.

The notebook focused on:

1. deterministic domain normalization for high-value entities,
2. chunk-aware BIO consistency handling,
3. alignment-aware evaluation to avoid misleading strict length filtering,
4. tokenizer-aware retraining and iterative sweeps when preprocessing-only gains plateau.

## What Was Innovative

### 1) Domain normalization layer

Implemented canonicalization rules for:

- email addresses,
- URLs,
- currency amounts,
- clock-like times,
- dates,
- prefixed IDs/SKUs.

Also added assertion checks to prevent prior false positives (for example, common words being mis-tagged as IDs and decimals being interpreted as times).

### 2) Alignment-aware scoring

Replaced strict token-count evaluability with token-text alignment coverage.

Impact:

- evaluable coverage increased from low strict-length behavior to high usable coverage,
- metrics are now computed on much larger aligned subsets, improving confidence in comparisons.

### 3) Decision-gated retraining workflow

Added a rule-driven gate:

- if inference-time gains remain near zero, trigger retraining,
- require at least +0.005 chunk F1 vs baseline before deployment recommendation.

### 4) Iterative retraining sweep

Added multi-configuration sweep logic with early stop on threshold success.

## Ablation Results (Inference-Time Variants)

From outputs/domain-tokenization-study/ablation_metrics.csv:

| Variant | Evaluable Rate | Chunk F1 | Accuracy | Mean Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.984 | 0.9562 | 0.9709 | 5.8967 |
| tokenization_plus_bio | 0.984 | 0.9562 | 0.9709 | 6.5883 |
| tokenization_only | 0.948 | 0.9063 | 0.9227 | 6.5385 |
| tokenization_bio_normalization | 0.948 | 0.9063 | 0.9227 | 6.0072 |

Interpretation:

- alignment-aware evaluability is strong (about 95% to 98%),
- tokenization+BIO repairs are quality-neutral relative to baseline,
- tokenization-only normalization variants degrade chunk quality in this setup.

## Decision Gate Output

From outputs/domain-tokenization-study/decision_gate.csv:

- best_variant: baseline
- baseline_f1: 0.9562
- best_f1: 0.9562
- f1_gain: 0.0000
- retraining_recommended: True

Reason emitted by notebook:

"F1 gain remains near zero after alignment-aware scoring; retraining with tokenizer-aware labels is the next step."

## Retraining and Sweep Outcomes

### Single retrain run

From outputs/domain-tokenization-study/retrain_metrics.csv:

- eval_f1: 0.8823
- this underperformed baseline significantly.

### Iterative sweep

From outputs/domain-tokenization-study/retrain_sweep.csv:

| Experiment | Status | Eval F1 | Gain vs Baseline |
| --- | --- | ---: | ---: |
| norm_off_e3_lr2e5_full | completed | 0.9582 | +0.0020 |
| norm_off_e3_lr1e5_full | completed | 0.9525 | -0.0036 |
| norm_off_e2_lr2e5_t6000 | completed | 0.9469 | -0.0093 |

Best sweep candidate improved over baseline, but not enough to pass deployment threshold (+0.005).

### Final retrain comparison

From outputs/domain-tokenization-study/retrain_comparison.csv:

- selected source: iterative_sweep
- selected experiment: norm_off_e3_lr2e5_full
- retrain_f1_gain: +0.0020
- deploy_retrained_model: False

## Downstream IE Impact

From outputs/domain-tokenization-study/ie_summary.csv:

- accepted_rate is 1.0 for all variants,
- mean confidence and role completeness are nearly unchanged,
- cost per accepted event changes are minimal.

Interpretation: downstream behavior is stable, but preprocessing innovation did not yield meaningful practical uplift yet.

## Final Innovation Assessment

This innovation work is successful in method and infrastructure, but not yet a deployment-quality model gain.

What was achieved:

1. stronger evaluation validity (alignment-aware coverage),
2. safer preprocessing rules with anti-regression assertions,
3. reproducible retraining/sweep pipeline with explicit decision thresholds.

Current recommendation:

- keep baseline for production,
- treat this notebook as an experimentation and governance upgrade,
- continue iteration with larger sweeps/model-scale trials if +0.005 F1 deployment gate remains the criterion.
