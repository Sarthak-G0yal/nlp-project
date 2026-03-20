# 12. Baseline Results (Logistic Regression)

## Reported Metrics

- token-level macro F1: 0.2678
- token-level weighted F1: 0.7293
- chunk-level overall F1 (seqeval): 0.6688

## Interpretation

- Weighted token F1 is reasonable because dominant labels are easier and frequent.
- Macro F1 is low, indicating weak performance on minority chunk classes.
- The baseline captures major patterns (NP/PP/VP) but fails on several sparse categories.

## Usefulness

This baseline serves as a meaningful control to quantify gains from contextual transformer modeling.
