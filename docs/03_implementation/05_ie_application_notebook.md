# 17. IE Application Notebook

Notebook: `Shallow_Parsing_IE_Application.ipynb`

## Purpose

This notebook demonstrates a practical application of shallow parsing for information extraction (IE), moving beyond benchmark metrics into usable extraction outputs.

It shows how chunk tags (NP/VP/PP) can be converted into structured records such as:

- subject phrases
- action phrases
- object phrases
- prepositional context phrases

## Why This Notebook Matters

The main training notebooks focus on model quality. This application notebook answers a different question:

- How can chunk predictions be used in an actual IE pipeline?

This is important for converting model output into business- or task-level value.

## Pipeline Implemented

1. Load CoNLL-2000 parquet data and decode chunk ids.
2. Convert BIO chunk tags into phrase spans (`bio_to_spans`).
3. Build sentence-level IE records (`sentence_to_ie_record`) with:
   - `subject`
   - `action`
   - `object`
   - `prep_phrases`
4. Run event-style filtering using a finance/news verb list.
5. Produce phrase analytics charts for top subjects and actions.
6. Optionally run custom inference from the trained transformer checkpoint.
7. Run an email event extraction pipeline from pasted email text.

## Core Heuristics Used

- Subject: first NP span
- Action: first VP span
- Object: first NP span after the first VP
- Context: all PP spans

These are intentionally simple and interpretable rules that can be extended later.

## Key Engineering Notes

The notebook includes safety handling for common runtime problems:

- robust parquet loading (`fastparquet` fallback behavior)
- stable DataFrame schema when event filtering returns zero rows
- automatic selection of latest trained checkpoint from:
  - `outputs/distilbert-conll2000/checkpoint-*`

This keeps the notebook usable even when training artifacts are organized by checkpoints.

## Example Output Type

For an input sentence like:

`The company announced a major acquisition in Europe yesterday.`

the notebook can produce a structured IE view with chunk-based labels and fields such as:

- subject: `The company`
- action: `announced`
- object: `a major acquisition`
- prep_phrases: `in Europe`

## New: Paste-Email Event Pipeline

The notebook now includes a dedicated section that lets you paste an email body and extract structured event rows.

### Input

- Free-form email text (multi-line), pasted into `email_text`.

### Processing Steps

1. Load or reuse the trained chunking pipeline checkpoint.
2. Split email content into candidate sentences (with basic header cleanup such as `Subject:`).
3. Predict chunk tags for each sentence.
4. Convert chunk outputs into IE tuples (subject/action/object).
5. Add event-level fields with lightweight heuristics:
  - event type
  - location phrase
  - time phrase

### Output Schema

The extracted DataFrame includes:

- `sentence`
- `event_type`
- `trigger`
- `subject`
- `object`
- `location`
- `time`
- `num_chunks`

### Typical Use

1. Open `Shallow_Parsing_IE_Application.ipynb`.
2. Go to the email extraction section.
3. Replace the sample `email_text` with your own email content.
4. Run the cell to get structured event records ready for downstream triage, alerts, or reporting.

## Limitations

- Rule-based subject/object extraction can fail on complex syntax.
- No coreference resolution (pronouns are unresolved).
- Event filtering relies on a hand-built verb list.
- For production use, chunking should be combined with NER and relation extraction.
- Location/time detection currently uses lightweight patterns and should be upgraded with dedicated temporal and geolocation models for higher precision.

## Suggested Extensions

1. Add confidence-aware extraction using model probabilities.
2. Add rule packs per domain (finance, legal, biomedical).
3. Link extracted tuples to entity normalization and KB entries.
4. Export IE records as JSONL/CSV for downstream systems.
