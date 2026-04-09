---
marp: true
theme: default
paginate: true
---

# Shallow Parsing Project (CoNLL-2000)
## Survey -> Implementation -> Analysis -> Innovation

This page gives the project map: objective, scope, and end-to-end delivery.

- Goal: build an end-to-end shallow parsing project for information extraction.
- Dataset: CoNLL-2000, 8,937 train and 2,013 test sentences.
- Scope: project delivery with reproducible engineering artifacts, not an instructional walkthrough.
- Deliverables: modeling pipeline, scaling study, innovation track, IE bridge, dashboard.
- Pipeline setup (project flow): data load -> schema check -> baseline and transformer modeling -> evaluation and error analysis -> scaling study -> innovation study -> IE bridge and dashboard.

Source: README.md, docs/01_project/01_overview.md, docs/03_implementation/01_data.md

---

## Survey: Existing Landscape in Shallow Parsing

This page summarizes the existing development trajectory in shallow parsing.

- Classical path: transformation-based learning, memory-based models, SVM/CRF with handcrafted features.
- Neural path: BiLSTM-CRF style sequence models.
- Current standard: transformer token classification with BIO labels and seqeval chunk metrics.
- Recent extensions: nonlinear CRF heads, span-aware approaches, and parameter-efficient tuning.

Source: docs/02_research/01_field_background.md, docs/02_research/02_recent_work.md

---

## Survey: Gap and Project Positioning

This page explains the research gap and exactly where this project fits.

- Gap in many resources: either baseline-only or transformer-only coverage.
- Missing in typical workflows: full comparison + reproducibility issues + deployment-oriented decisions.
- Project position: one integrated workflow with baseline, transformer, scaling, innovation, and IE application.
- Project novelty: practical, reproducible implementation and decision framework rather than new architecture.

Source: docs/02_research/03_gap_and_project_positioning.md, docs/01_project/03_why_this_project.md

---

## Implementation: What Existed vs What We Implemented

This page maps prior work to concrete components that were implemented in this project.

| Existing development in field | Implemented in this project |
| --- | --- |
| CoNLL-2000 chunking benchmark | Robust parquet data pipeline with validation and decoding |
| Classical baselines | POS-feature Logistic Regression baseline |
| Transformer sequence labeling | DistilBERT token classifier (HF Trainer + seqeval) |
| Model comparison studies | Multi-notebook benchmark summary (NB, Logistic, BiLSTM, CRF, Transformer) |
| Production-facing demos | Chunk-to-event bridge + Streamlit dashboard + email pipeline |

Pipeline setup (implementation):
- Data track: train and test parquet -> schema checks -> chunk label decoding.
- Model track A: token and POS features -> Logistic Regression baseline.
- Model track B: tokenizer and word-label alignment -> DistilBERT fine-tuning.
- Evaluation track: token metrics + seqeval chunk metrics + error analysis slices.

Source: docs/03_implementation/01_data.md, docs/03_implementation/02_notebook_pipeline.md, docs/03_implementation/03_modeling_decisions.md, docs/03_implementation/07_streamlit_dashboard.md

---

## Implementation: Core Model Outcomes

This page reports the main quantitative gains from implemented models.

- Baseline Logistic Regression chunk F1: 0.6688.
- DistilBERT chunk F1: 0.9586 (absolute +0.2898 vs baseline).
- Relative gain: +43.33%; chunk error reduction: 87.49%.
- DistilBERT also leads over additional baselines in this project setup:
  CRF 0.9307, Logistic 0.8975, MultinomialNB 0.8726, BiLSTM 0.8025.

$$
\Delta F1 = F1_{\text{DistilBERT}} - F1_{\text{baseline}} = 0.9586 - 0.6688 = 0.2898
$$

$$
\mathrm{RelativeGain}(\%) = \frac{\Delta F1}{F1_{\mathrm{baseline}}} \times 100 = 43.33\%
$$

$$
\mathrm{ErrorReduction}(\%) = \frac{(1 - F1_{\mathrm{baseline}}) - (1 - F1_{\mathrm{DistilBERT}})}{1 - F1_{\mathrm{baseline}}} \times 100 = 87.49\%
$$

Source: docs/04_results/01_baseline_results.md, docs/04_results/02_transformer_results.md, docs/04_results/04_notebook_results_summary.md

---

## Analysis: Performance and Error Findings

This page interprets what the metrics mean for model behavior and downstream extraction quality.

- Baseline token macro F1 is low (0.2678): weak minority-class behavior.
- Transformer substantially improves boundary consistency and rare-class robustness.
- Remaining errors are mainly rare labels and ambiguous boundary contexts.
- Practical IE impact: better chunk boundaries reduce noisy extracted spans downstream.

$$
F1 = \frac{2PR}{P + R}
$$

Where $P$ is precision and $R$ is recall; improvements in both reduced boundary mistakes that affect extraction quality.

Source: docs/04_results/01_baseline_results.md, docs/04_results/03_error_analysis.md

---

## Analysis: Scaling Study ROI (Is Extra Compute Worth It?)

This page evaluates the compute-versus-quality trade-off and gives a deployment decision.

- Scaling setup (same run config): DistilBERT, BERT-base, TinyBERT, NanoBERT-like.
- DistilBERT: F1 0.9563, train 120.0361s, 66.3806M params.
- BERT-base: F1 0.9602, gain +0.0039, but 1.8650x training time and 1.6407x params.
- Decision threshold in project: minimum +0.005 F1 gain.
- Decision: extra training compute is not worth it under current threshold; keep DistilBERT as default quality-cost choice.

$$
\mathrm{ComputeROI} = \frac{\Delta F1}{\mathrm{TrainTimeRatio}} = \frac{0.0039}{1.8650} \approx 0.0021
$$

$$
\Delta F1 \ge 0.005
$$

Source: docs/04_results/05_larger_transformer_comparison.md

---

## Innovation: Domain-Aware Preprocessing Track

This page shows the innovation design and ablation outcomes for domain-aware preprocessing.

- Innovation introduced: normalization rules for email, URL, currency, time/date, and ID-like patterns.
- Added alignment-aware evaluability to avoid strict length-only filtering artifacts.
- Ablation results (chunk F1):
  baseline 0.9562,
  tokenization+BIO 0.9562,
  tokenization-only 0.9063,
  tokenization+BIO+normalization 0.9063.
- Coverage remained high with alignment-aware scoring (evaluable rate 0.948 to 0.984).

$$
\mathrm{EvaluableRate} = \frac{N_{\mathrm{evaluable}}}{N_{\mathrm{total}}}
$$

Source: docs/04_results/07_domain_specific_tokenization_innovation.md, outputs/domain-tokenization-study/ablation_metrics.csv

---

## Innovation: Retraining and IE Bridge Outcomes

This page reports retraining gate decisions and practical IE pipeline outputs.

- Retraining sweep best run: F1 0.9582, gain +0.0020 vs baseline.
- Project deployment gate: require >= +0.005 F1 for replacement.
- Decision: deploy_retrained_model = False (innovation infra is valuable, gain still below deployment bar).
- IE bridge and application are implemented end-to-end:
  email demo accepted events 5/5, total cost estimate 38.00, cost per accepted event 7.60.

Pipeline setup (IE bridge):
- Email text -> sentence split -> chunk prediction -> BIO spans -> event schema mapping -> confidence and abstention -> cost summary.

$$
\mathrm{Deploy} = \mathbf{1}[\Delta F1 \ge 0.005]
$$

Source: outputs/domain-tokenization-study/retrain_sweep.csv, outputs/domain-tokenization-study/retrain_comparison.csv, docs/04_results/06_chunk_to_event_bridge_results.md

---

## Final Slide: Project Contributions and Decisions

This page closes the story with final project contributions and decisions.

- Survey: mapped field development from classical chunking to transformer-era practice.
- Implementation: built reproducible pipeline, baseline and transformer training, dashboard, and event bridge.
- Analysis: validated strong transformer gains and quantified compute-quality trade-off.
- Innovation: delivered domain-aware preprocessing and retraining governance pipeline, but no deployment-level gain yet.
- Final model choice now: DistilBERT baseline for deployment; continue innovation/scaling only if gains exceed +0.005 gate.

$$
\mathrm{KeepDistilBERT} \Leftarrow \Delta F1_{\mathrm{candidate}} < 0.005
$$

Source: docs/README.md, docs/04_results/04_notebook_results_summary.md, docs/04_results/05_larger_transformer_comparison.md, docs/04_results/07_domain_specific_tokenization_innovation.md, docs/05_conclusion/01_conclusions.md