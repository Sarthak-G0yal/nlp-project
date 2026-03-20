# 9. Notebook Pipeline

The notebook is divided into sequential phases.

## Phase 0

- imports and global path setup

## Phase 1

- robust parquet loading (`fastparquet` first, `pyarrow` fallback)
- schema checks

## Phase 2

- decode POS/chunk label ids into readable labels

## Phase 3

- exploratory analysis for sequence lengths and label frequency

## Phase 4

- POS-only baseline using scikit-learn logistic regression
- token-level metrics and optional seqeval chunk-level metrics

## Phase 5

- Hugging Face dataset conversion
- tokenization and word-label alignment
- Trainer setup for transformer fine-tuning
- training and final evaluation

## Phase 6

- error analysis utilities
