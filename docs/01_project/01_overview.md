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

- `Notebook.ipynb`: end-to-end data, modeling, training, and evaluation workflow
- `dataset/train.parquet`, `dataset/test.parquet`: prepared CoNLL-2000 data files
- `outputs/distilbert-conll2000/`: training checkpoints and evaluation artifacts
