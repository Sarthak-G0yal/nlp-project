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
