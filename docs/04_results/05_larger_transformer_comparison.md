# Larger Transformer Comparison Results

## Objective

This experiment measured whether moving from a compact transformer (DistilBERT) to larger encoders (BERT-base, RoBERTa-base) provides enough chunking quality gain to justify extra training cost, while also testing compact tiny-model candidates.

Notebook used: `ipynb/Larger_Transformer_Comparison.ipynb`

## What I Did

1. Loaded CoNLL-2000 train and test splits from parquet.
2. Trained token-classification models with the same pipeline and metric function (seqeval chunk metrics).
3. Compared the primary measured encoders:
   - distilbert/distilbert-base-uncased
   - bert-base-uncased
   - roberta-base
4. Added lightweight candidates in config for extended scaling:
   - google/bert_uncased_L-2_H-128_A-2 (NanoBERT-like)
   - prajjwal1/bert-tiny
   - huawei-noah/TinyBERT_General_4L_312D
5. Used identical run configuration for fairness:
   - train samples: 8937
   - test samples: 2013
   - epochs: 2
   - batch size: 16 (train/eval)
   - learning rate: 2e-5
6. Computed cost/benefit analysis relative to DistilBERT baseline:
   - absolute F1 gain
   - training-time multiplier
   - parameter-count multiplier
   - gain per extra training hour

## Expanded Candidate Pool (Current State)

| Model | Status |
| --- | --- |
| distilbert/distilbert-base-uncased | measured |
| bert-base-uncased | measured |
| roberta-base | failed (CUDA OOM in this run) |
| google/bert_uncased_L-2_H-128_A-2 | measured |
| prajjwal1/bert-tiny | failed (tokenizer backend dependency) |
| huawei-noah/TinyBERT_General_4L_312D | measured |

## Results

| Model | Params (M) | Train Time (s) | Chunk F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bert-base-uncased | 108.9093 | 223.8648 | 0.9602 | 0.9582 | 0.9622 | 0.9751 |
| distilbert/distilbert-base-uncased | 66.3806 | 120.0361 | 0.9563 | 0.9531 | 0.9596 | 0.9727 |
| huawei-noah/TinyBERT_General_4L_312D | 14.2598 | 32.7503 | 0.8806 | 0.8750 | 0.8863 | 0.9339 |
| google/bert_uncased_L-2_H-128_A-2 | 4.3724 | 17.1686 | 0.7405 | 0.7213 | 0.7608 | 0.8465 |

### Failed Runs (This Execution)

- roberta-base: failed with CUDA out-of-memory on the current GPU setup.
- prajjwal1/bert-tiny: failed tokenizer backend initialization; requires `sentencepiece` or `tiktoken` in this environment.

## Relative to DistilBERT Baseline

| Model | F1 Gain vs Base | Train Time Ratio | Params Ratio | Gain per Extra Hour |
| --- | ---: | ---: | ---: | ---: |
| bert-base-uncased | +0.0039 | 1.8650x | 1.6407x | 0.1344 |
| huawei-noah/TinyBERT_General_4L_312D | -0.0757 | 0.2728x | 0.2148x | n/a |
| google/bert_uncased_L-2_H-128_A-2 | -0.2158 | 0.1430x | 0.0659x | n/a |

## Decision Rule Used

- Minimum meaningful F1 gain: 0.005
- Maximum allowed training-time multiplier: 2.0

Outcome from notebook recommendation cell:
- DistilBERT is the baseline.
- Recommendation: bigger models are not clearly worth it under the current threshold for this run.
- BERT-base is only +0.0039 over DistilBERT, below the +0.005 threshold.
- Tiny models that completed are much faster but substantially lower quality.

## Interpretation

- DistilBERT remains the best quality-efficiency balance in this execution.
- BERT-base gives the highest measured F1 here, but the gain over DistilBERT is marginal relative to extra cost.
- TinyBERT and NanoBERT-like are very compute-efficient but lose substantial chunk quality.

## Recommended Usage

- Use DistilBERT for default deployment when balancing quality and cost.
- Use BERT-base only when squeezing small extra quality is worth nearly 1.87x training time.
- Use TinyBERT or NanoBERT-like only under strict runtime/resource constraints.
- To complete the frontier cleanly, fix `prajjwal1/bert-tiny` dependencies and rerun `roberta-base` on a larger-memory setup or reduced batch configuration.