# Larger Transformer Comparison Results

## Objective

This experiment measured whether moving from a compact transformer (DistilBERT) to larger encoders (BERT-base, RoBERTa-base) provides enough chunking quality gain to justify extra training cost.

Notebook used: Larger_Transformer_Comparison.ipynb

## What We Did

1. Loaded CoNLL-2000 train and test splits from parquet.
2. Trained token-classification models with the same pipeline and metric function (seqeval chunk metrics).
3. Compared three encoders:
   - distilbert/distilbert-base-uncased
   - bert-base-uncased
   - roberta-base
4. Used identical run configuration for fairness:
   - train samples: 8937
   - test samples: 2013
   - epochs: 2
   - batch size: 16 (train/eval)
   - learning rate: 2e-5
5. Computed cost/benefit analysis relative to DistilBERT baseline:
   - absolute F1 gain
   - training-time multiplier
   - parameter-count multiplier
   - gain per extra training hour

## Results

| Model | Params (M) | Train Time (s) | Chunk F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| roberta-base | 124.0727 | 224.6860 | 0.9665 | 0.9654 | 0.9675 | 0.9780 |
| bert-base-uncased | 108.9093 | 222.7203 | 0.9602 | 0.9582 | 0.9622 | 0.9751 |
| distilbert/distilbert-base-uncased | 66.3806 | 120.0704 | 0.9566 | 0.9534 | 0.9598 | 0.9730 |

## Relative to DistilBERT Baseline

| Model | F1 Gain vs Base | Train Time Ratio | Params Ratio | Gain per Extra Hour |
| --- | ---: | ---: | ---: | ---: |
| roberta-base | +0.0099 | 1.8713x | 1.8691x | 0.3397 |
| bert-base-uncased | +0.0036 | 1.8549x | 1.6407x | 0.1265 |

## Decision Rule Used

- Minimum meaningful F1 gain: 0.005
- Maximum allowed training-time multiplier: 2.0

Outcome from notebook recommendation cell:
- DistilBERT is the baseline.
- RoBERTa-base is worth considering under this threshold.
- BERT-base does not clear the minimum gain threshold.

## Interpretation

- RoBERTa-base produced the best quality and a meaningful gain over DistilBERT (+0.99 F1 points) while staying under the 2.0x training-time budget.
- BERT-base improved over DistilBERT, but the gain (+0.36 F1 points) is likely too small for nearly the same compute increase as RoBERTa.
- DistilBERT remains the strongest efficiency choice when latency and training budget are primary constraints.

## Recommended Usage

- Use DistilBERT for fast iteration and lower compute budgets.
- Use RoBERTa-base when you can afford ~1.87x training cost for higher chunking quality.
- Skip BERT-base in this specific setup unless there are deployment constraints that make it preferable to RoBERTa.