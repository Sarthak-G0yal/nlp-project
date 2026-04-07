# 19. Streamlit Dashboard Library

App path: `libs/chunk_event_dashboard/app.py`

## Purpose

This module provides an interactive dashboard for running local chunking checkpoints and viewing chunk-to-event extraction outputs without opening notebooks.

It is implemented as a separate library folder under `libs/` and managed through the repository-level `uv` environment.

## What Is Implemented

- model checkpoint discovery from `outputs/`
- checkpoint-aware model selection in the UI
- single-text extraction mode:
  - token-level chunk predictions
  - chunk spans
  - structured event record
- email pipeline mode:
  - sentence splitting and event extraction
  - confidence and abstention fields
  - cost proxy summary and CSV download

## Run Commands

### From repository root

```bash
uv sync
uv run streamlit run libs/chunk_event_dashboard/app.py --global.dataFrameSerialization legacy
```

### From dashboard folder

```bash
cd libs/chunk_event_dashboard
uv run streamlit run app.py
```

The dashboard folder includes a local Streamlit config at `.streamlit/config.toml` with legacy dataframe serialization enabled.

## Runtime Compatibility Notes

The current stack includes Streamlit 1.19 with newer pandas/pyarrow versions. Two compatibility fixes are applied:

1. Arrow table compatibility:
   - avoid frontend decode errors such as `Unrecognized type: LargeUtf8`
   - use legacy Streamlit dataframe serialization
2. Legacy chart dtype compatibility:
   - avoid `NotImplementedError: Dtype str not understood`
   - render event distribution with matplotlib (`st.pyplot`) instead of `st.bar_chart`

## Main Library Files

- `libs/chunk_event_dashboard/src/chunk_event_dashboard/constants.py`
- `libs/chunk_event_dashboard/src/chunk_event_dashboard/inference.py`
- `libs/chunk_event_dashboard/src/chunk_event_dashboard/extraction.py`
- `libs/chunk_event_dashboard/app.py`

## Troubleshooting

If an old error still appears after code updates:

1. stop all existing Streamlit processes
2. relaunch using one of the commands above
3. hard-refresh the browser tab (Ctrl+Shift+R)
