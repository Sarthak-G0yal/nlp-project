# 5. Recent Research and Trends

## Transformer-Based Token Classification

Fine-tuning pretrained language models for token classification is the dominant recipe in modern NLP practice. This approach typically outperforms feature-based baselines and is easy to adapt to different labeling schemes.

## Beyond Linear-Chain CRF

Recent work such as Locally-Contextual Nonlinear CRFs (Shah et al., 2021) shows gains on sequence labeling, including strong chunking results on CoNLL-2000, by modeling richer local context in potential functions.

## Span-Oriented and Hybrid Formulations

There is increased use of span-based modeling and hybrid architectures in related extraction tasks, especially when boundary quality is critical.

## Retrieval-Oriented Chunking (Adjacent Trend)

Recent ACL 2025 works (for example MoC and AutoChunker) focus on document chunking for retrieval-augmented generation. This is adjacent to but not equivalent to classic syntactic shallow parsing. It still informs practical chunking design decisions.

## Project Implementation of the Research Direction

This project operationalizes the modern direction through:

- transformer token-classification fine-tuning
- seqeval chunk-level evaluation
- direct comparison with a lightweight statistical baseline
