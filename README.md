# Shallow Parsing for Information Extraction (CoNLL-2000)

Detailed project documentation is available in `docs/README.md`.

This repository contains an end-to-end shallow parsing (chunking) project using CoNLL-2000 data from `dataset/train.parquet` and `dataset/test.parquet`.

Core project artifacts include:

- `ipynb/Notebook.ipynb`: main baseline-vs-transformer workflow
- `ipynb/Other_Shallow_Parsing_Baselines.ipynb`: additional baseline comparison set (CRF, MultinomialNB, LogisticRegression, BiLSTM)
- `ipynb/Larger_Transformer_Comparison.ipynb`: scaling study across compact and larger encoders
- `ipynb/Domain_Specific_Tokenization_Chunk_Aware_Preprocessing.ipynb`: domain-aware preprocessing and retraining gate
- `ipynb/Shallow_Parsing_IE_Application.ipynb`: chunk-to-IE application flow
- `ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb`: confidence-aware event bridge and cost policy layer

## Problem Statement

Shallow parsing (text chunking) segments a sentence into non-overlapping phrase chunks such as NP, VP, and PP. In information extraction, chunking provides phrase boundaries that improve downstream extraction quality and interpretability.

## What Is Latest In This Field

Recent progress in shallow parsing and closely related sequence labeling has followed these trends:

- Transformer-based token classification as the default strong baseline:
	Fine-tuning pretrained encoders (for example BERT/DistilBERT/DeBERTa-family) with BIO/BIOES labels is now standard practice for chunking-like tasks.

- Improved sequence modeling heads over plain linear-chain CRF:
	Locally-Contextual Nonlinear CRFs (Shah et al., 2021) report improvements in sequence labeling and specifically strong CoNLL-2000 chunking performance compared with earlier CRF heads.

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

## Outputs and Experiment Artifacts

Current outputs are organized under `outputs/`:

- `outputs/distilbert-conll2000/`: primary model checkpoints
- `outputs/scale-study-bert-base-uncased/`, `outputs/scale-study-distilbert-base-uncased/`, `outputs/scale-study-roberta-base/`: scaling comparison checkpoints
- `outputs/domain-tokenization-study/`: innovation-study CSV artifacts (`ablation_metrics.csv`, `decision_gate.csv`, `retrain_sweep.csv`, and related analysis tables)
- `outputs/domain-tokenization-retrain-distilbert/` and `outputs/domain-tokenization-retrain-sweep/`: retraining trial checkpoints

## Environment and Dependencies

The project environment is managed with `uv`.

- dependency specification: `pyproject.toml`
- lockfile: `uv.lock`
- compatibility export: `requirements.txt`

Reproducible environment sync:

```bash
uv sync
```

## Expected Deliverables

- Reproducible training/evaluation notebook
- Baseline and transformer model comparison (Precision/Recall/F1)
- Error slices by chunk class (for example NP vs VP boundaries)
- Project summary on practical IE takeaways from shallow parsing

## Streamlit Dashboard (Separate Library)

An interactive dashboard is available in:

- `libs/chunk_event_dashboard/app.py`

The dashboard provides:

- model checkpoint selection from local `outputs/`
- single text chunk-to-event extraction
- email event pipeline with confidence and cost summary

To deploy from the repository root:

```bash
uv sync
uv run streamlit run libs/chunk_event_dashboard/app.py --global.dataFrameSerialization legacy
```

For implementation details and troubleshooting, see:

- `docs/03_implementation/07_streamlit_dashboard.md`


## Next Steps
- compare currently evaluated baselines and transformer models across all metrics.
- expand model families and optimization settings for stronger scaling conclusions.
- continue improving shallow parsing IE application quality and event extraction reliability.
- evaluate whether larger transformer compute cost is justified by measurable gains.