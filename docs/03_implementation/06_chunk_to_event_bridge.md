# 18. Chunk-to-Event Bridge Notebook

Notebook: `ipynb/Chunk_to_Event_Cost_Aware_Bridge.ipynb`

## Purpose

This notebook implements a bridge layer that converts shallow chunk predictions into structured event records and combines them with cost-aware model selection.

It extends the IE application flow with three additional ideas:

1. explicit event schema and role requirements
2. confidence-based abstention
3. compute-aware model recommendation

## What Is Implemented

### Event Schema Layer

- keyword-driven event types: meeting, deadline, travel, incident, announcement, other
- role requirement map per event type (trigger, time, location, subject)
- sentence-level extraction of location and time phrases

### Chunk-to-Event Conversion

- `bio_to_spans` converts BIO chunk labels into phrase spans
- `sentence_to_ie_record` maps spans to subject/action/object candidates
- `extract_event_record` builds the final event record with:
  - event type
  - trigger
  - subject
  - object
  - location
  - time
  - confidence
  - abstained flag and reason

### Confidence and Abstention

Confidence combines:

- mean token confidence from model output
- role completeness score
- lexical bonus for non-`other` event classes

Records below threshold are marked as abstained.

### Cost-Aware Model Layer

- model profile table includes measured models and pending tiny models
- quality-first and latency-first recommendation policy
- pending models are handled explicitly as `pending_evaluation` (no pipeline break)

Latest scaling sync state (outside the bridge notebook profile cell):

- measured: `google/bert_uncased_L-2_H-128_A-2`, `huawei-noah/TinyBERT_General_4L_312D`
- failed in latest run: `roberta-base` (CUDA OOM), `prajjwal1/bert-tiny` (tokenizer backend dependency)
- bridge notebook profile snapshot still needs rerun to ingest those measured tiny-model rows

### Email Pipeline

- parses multi-line email input
- removes common header lines
- extracts sentence-level event records
- tracks proxy event cost and cost-per-accepted-event

## Current Status

- The notebook is executable and currently run through the implemented cells.
- Core bridge outputs are available for:
  - measured model profile comparison
  - recommendation candidates
  - email event extraction with confidence and cost estimates

## Limitations

1. Event scoring is still heuristic and not calibrated on labeled event data.
2. Cost uses proxy units derived from parameter scale; not yet direct measured latency/token.
3. Bridge model-profile cells can lag behind standalone scaling runs unless refreshed.

## Next Implementation Targets

1. Add calibrated confidence (temperature scaling or validation-based calibration).
2. Replace proxy cost with measured inference latency and memory profiles.
3. Add event-level precision/recall/F1 against labeled event annotations.
4. Export accepted events to JSONL/CSV for downstream use.