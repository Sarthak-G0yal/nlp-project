# 11. Engineering Issues and Fixes

This project encountered multiple environment/runtime issues and resolved them in-notebook.

## Issue 1: Parquet Arrow Extension Conflict

Error symptom:

- `ArrowKeyError: pandas.period already defined`

Fix:

- implemented `read_parquet_safe()`
- load with `fastparquet` first, fallback to `pyarrow`

## Issue 2: scikit-learn LogisticRegression API Change

Error symptom:

- unexpected argument `multi_class`

Fix:

- removed deprecated/unsupported `multi_class` argument for installed sklearn version

## Issue 3: seqeval JSON Serialization

Error symptom:

- `int64 is not JSON serializable`

Fix:

- recursive conversion helper from numpy scalar objects to native Python values before JSON printing

## Issue 4: Trainer + Accelerate Compatibility

Error symptom:

- `Accelerator.unwrap_model() got an unexpected keyword argument 'keep_torch_compile'`

Fix:

- added compatibility shim in notebook to support environments where `unwrap_model` lacks this argument

## Issue 5: Notebook Callback Evaluation RuntimeError

Error symptom:

- `on_train_begin must be called before on_evaluate`

Fix:

- removed notebook progress callback prior to evaluation
- added logic to skip retraining if the trainer already has a trained state
