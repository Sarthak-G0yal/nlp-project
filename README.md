# Shallow Parsing for Information Extraction (CoNLL-2000)

Detailed project documentation is available in `docs/README.md`.

This project builds a complete shallow parsing (chunking) pipeline using the CoNLL-2000 dataset from `dataset/train.parquet` and `dataset/test.parquet`.

The notebook covers:
- dataset inspection and BIO chunk decoding
- exploratory analysis of chunk distribution
- a lightweight baseline (POS-based token classifier)
- a modern transformer token-classification setup for chunking
- detailed error analysis for information extraction quality

## 1) Problem Statement

Shallow parsing (text chunking) segments a sentence into non-overlapping phrase chunks such as NP, VP, and PP. In information extraction, chunking provides phrase boundaries that improve downstream extraction quality and interpretability.

## 2) What Is Latest In This Field

Recent progress in shallow parsing and closely related sequence labeling has followed these trends:

- Transformer-based token classification as the default strong baseline:
	Fine-tuning pretrained encoders (for example BERT/DistilBERT/DeBERTa-family) with BIO/BIOES labels is now standard practice for chunking-like tasks.

- Improved sequence modeling heads over plain linear-chain CRF:
	Locally-Contextual Nonlinear CRFs (Shah et al., 2021) report improvements on sequence labeling and specifically strong CoNLL-2000 chunking performance compared with earlier CRF heads.

- Span-centric formulations:
	Modern libraries and models increasingly use span-aware methods for token classification and extraction, often improving robustness on boundary decisions.

- Parameter-efficient adaptation:
	Adapter/LoRA-style tuning is increasingly used when full fine-tuning is costly.

- Retrieval/document chunking wave (2024-2025):
	ACL 2025 includes works such as MoC and AutoChunker focused on retrieval-oriented text chunking. This is related but different from classic syntactic shallow parsing; still useful for project discussion as an adjacent trend.

Reference points:
- CoNLL-2000 shared task paper (Tjong Kim Sang and Buchholz, 2000)
- Locally-Contextual Nonlinear CRFs for Sequence Labeling (arXiv:2103.16210)
- Hugging Face token-classification training recipe (current practical standard)

## 3) Notebook Workflow

The notebook is organized into phases:

1. Load parquet files and validate schema (`id`, `tokens`, `pos_tags`, `chunk_tags`).
2. Decode numeric POS/chunk IDs to readable labels.
3. Run EDA and inspect class imbalance.
4. Train a POS-only baseline using scikit-learn.
5. Prepare Hugging Face dataset objects for transformer fine-tuning.
6. Define and train a token-classification model with seqeval metrics.
7. Evaluate and perform error analysis by chunk type.

## 4) Dependencies You Need To Install

Current environment already has `pandas`, `numpy`, and parquet support.

Please install these additional packages for full notebook execution:

- `scikit-learn`
- `datasets`
- `transformers`
- `evaluate`
- `seqeval`
- `torch`
- `matplotlib`
- `seaborn`
- `tqdm`

Suggested install command (if you want):

```bash
uv add scikit-learn datasets transformers evaluate seqeval torch matplotlib seaborn tqdm
```

## 5) Expected Deliverables

- Reproducible training/evaluation notebook
- Baseline and transformer model comparison (Precision/Recall/F1)
- Error slices by chunk class (for example NP vs VP boundaries)
- Project summary on practical IE takeaways from shallow parsing

## 6) Streamlit Dashboard (Separate Library)

An interactive dashboard is available in:

- `libs/chunk_event_dashboard/app.py`

It supports:

- model checkpoint selection from local `outputs/`
- single text chunk-to-event extraction
- email event pipeline with confidence and cost summary

Run from repository root:

```bash
uv sync
uv run streamlit run libs/chunk_event_dashboard/app.py --global.dataFrameSerialization legacy
```

For implementation details and troubleshooting, see:

- `docs/03_implementation/07_streamlit_dashboard.md`


## Next Steps:
- compare models that are used till now ( baseline and all transformer) compare on all metrics.
- Implement other models or methods.
- implement a application of shallow pharsing for IE etc.
- use even bigger transformer model and see if the extra compute justify the gains