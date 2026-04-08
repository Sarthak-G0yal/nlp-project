# 10. Modeling Decisions and Rationale

## Baseline Model

Model: Logistic Regression on sparse token/POS feature vectors.

Why:

- fast to train
- interpretable reference point
- useful lower bound for modern model comparison

Feature set:

- lowercased token
- current POS id
- previous and next POS id
- simple orthographic flags (titlecase, uppercase, digit)

## Transformer Model

Model: DistilBERT token classification head.

Why:

- strong practical baseline in modern sequence labeling
- lower compute than larger transformer families
- straightforward Hugging Face integration

Key settings:

- learning rate: 2e-5
- epochs: 3
- batch size: 16 (train/eval)
- metric: seqeval F1 for model selection

## Scaling Study Model Pool

For compute-vs-gain analysis, the project uses an expanded candidate set in `ipynb/Larger_Transformer_Comparison.ipynb`.

Latest measured models in the current scaling execution:

- distilbert/distilbert-base-uncased
- bert-base-uncased
- google/bert_uncased_L-2_H-128_A-2 (NanoBERT-like compact variant)
- huawei-noah/TinyBERT_General_4L_312D

Configured candidates with unresolved run issues in the latest execution:

- roberta-base (CUDA OOM in this environment)
- prajjwal1/bert-tiny

`prajjwal1/bert-tiny` currently requires tokenizer backend dependency support (`sentencepiece` or `tiktoken`) for this setup.

Rationale:

- preserve a strong quality frontier with larger models
- explicitly test low-cost model options for budget-constrained deployment
