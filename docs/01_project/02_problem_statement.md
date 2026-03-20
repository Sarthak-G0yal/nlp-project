# 2. Problem Statement

## What is Shallow Parsing?

Shallow parsing (chunking) identifies non-overlapping phrase chunks in a sentence without building a full parse tree. Typical chunk types include:

- Noun Phrases (NP)
- Verb Phrases (VP)
- Prepositional Phrases (PP)

## Why It Matters for Information Extraction

Information extraction systems often need phrase boundaries to:

- extract entities and arguments with cleaner spans
- reduce noise compared with token-only tagging
- provide interpretable intermediate structure for downstream modules

## Task Formulation in This Project

Given a token sequence and corresponding POS tags, predict one chunk label per token from the CoNLL-2000 label set.

We evaluate with:

- token-level precision/recall/F1
- chunk-level seqeval metrics

## Success Criteria

- build a reproducible pipeline from data to model evaluation
- establish a baseline model for comparison
- train a modern transformer model and compare gains
- analyze errors by chunk category and confusion pattern
