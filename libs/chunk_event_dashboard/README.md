# Chunk Event Dashboard

Separate library-style dashboard for the CoNLL-2000 shallow parsing project.

## What it does

- loads local token-classification checkpoints from `outputs/`
- takes free-text or email input
- predicts chunk tags
- converts chunks to event records with confidence/abstention
- shows event cost proxy based on notebook model profile

## Run (from repository root)

```bash
uv sync
uv run streamlit run libs/chunk_event_dashboard/app.py --global.dataFrameSerialization legacy
```

## Run (from dashboard folder)

```bash
cd libs/chunk_event_dashboard
uv run streamlit run app.py
```

This folder includes `.streamlit/config.toml` with legacy dataframe serialization enabled for Streamlit 1.19 compatibility.

## Troubleshooting

### Error: `Unrecognized type: "LargeUtf8"`

Cause:

- Streamlit 1.19 frontend Arrow decoding can fail with newer pandas/pyarrow string serialization.

Fix:

- run with legacy dataframe serialization (already configured in this folder)
- if launching from repo root, include:

```bash
uv run streamlit run libs/chunk_event_dashboard/app.py --global.dataFrameSerialization legacy
```

### Error: `NotImplementedError: Dtype str not understood`

Cause:

- legacy Streamlit chart marshalling path can fail on pandas string dtype in `st.bar_chart`.

Fix:

- app uses matplotlib (`st.pyplot`) for event distribution chart instead of `st.bar_chart`

### If stale errors still show

1. stop all running Streamlit processes
2. relaunch the app
3. hard-refresh browser tab (`Ctrl+Shift+R`)

## Library layout

- `libs/chunk_event_dashboard/src/chunk_event_dashboard/constants.py`
- `libs/chunk_event_dashboard/src/chunk_event_dashboard/inference.py`
- `libs/chunk_event_dashboard/src/chunk_event_dashboard/extraction.py`
- `libs/chunk_event_dashboard/app.py`
