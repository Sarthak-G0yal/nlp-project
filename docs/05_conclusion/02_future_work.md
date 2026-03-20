# 16. Future Work

## Modeling

- evaluate larger encoders (for example RoBERTa/DeBERTa)
- compare CRF and nonlinear CRF heads on top of transformer embeddings
- explore parameter-efficient tuning (LoRA/adapters)

## Evaluation

- report per-label confidence calibration
- add cross-validation or repeated runs for robustness
- evaluate domain transfer behavior on out-of-domain corpora

## Deployment and Usability

- export a lightweight inference pipeline for production use
- add CLI/API wrappers for batch chunk prediction
- create automated tests for label mapping and metric integrity
