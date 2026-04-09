# 1. Project Overview

## Title

Shallow Parsing for Information Extraction using CoNLL-2000.

## One-Line Summary

I built and evaluated phrase-level chunking models, from a lightweight POS-feature baseline to a transformer-based token classifier, to study practical information extraction performance.

## Scope

This project focuses on syntactic shallow parsing (text chunking), not full parsing.

Input:
- sentence tokens
- POS tags

Output:
- chunk tags in BIO-style format (for example B-NP, I-NP, B-VP, O)

## Main Artifacts

- `ipynb/Notebook.ipynb`: end-to-end data, modeling, training, and evaluation workflow
- `ipynb/Other_Shallow_Parsing_Baselines.ipynb`: extended non-transformer baseline benchmark
- `ipynb/Larger_Transformer_Comparison.ipynb`: scaling experiment across compact and larger encoders
- `ipynb/Domain_Specific_Tokenization_Chunk_Aware_Preprocessing.ipynb`: innovation notebook with retraining gate outputs
- `ipynb/Shallow_Parsing_IE_Application.ipynb` and `ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb`: IE application and event-bridge notebooks
- `dataset/train.parquet`, `dataset/test.parquet`: prepared CoNLL-2000 data files
- `outputs/distilbert-conll2000/`: training checkpoints and evaluation artifacts
- `outputs/domain-tokenization-study/`: persisted CSV summaries for innovation ablations and retraining decisions
- `libs/chunk_event_dashboard/app.py`: interactive dashboard entry point
