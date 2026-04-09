from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_string_dtype

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunk_event_dashboard.constants import MODEL_COST_PROFILE
from chunk_event_dashboard.extraction import extract_event_record, run_email_event_pipeline
from chunk_event_dashboard.inference import (
    available_model_checkpoints,
    find_project_root,
    load_token_classifier,
    read_checkpoint_metrics,
)


st.set_page_config(page_title="Chunk-to-Event Dashboard", page_icon="NLP", layout="wide")

# Streamlit 1.19 can fail on Arrow LargeUtf8 with newer pandas/pyarrow combos.
try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def get_pipeline(model_key: str, project_root_text: str):
    project_root = Path(project_root_text)
    return load_token_classifier(model_key, project_root)


def streamlit_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string-like columns/index to object for broader Streamlit compatibility."""
    safe_df = df.copy()
    for col in safe_df.columns:
        if is_string_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].astype("object")

    if hasattr(safe_df.index, "dtype") and is_string_dtype(safe_df.index.dtype):
        safe_df.index = safe_df.index.astype("object")

    return safe_df


def _lookup_profile_row(profile_lookup: dict, model_name: str) -> dict:
    row = profile_lookup.get(model_name)
    if row:
        return row

    short_name = model_name.split("/")[-1]
    for key, candidate in profile_lookup.items():
        if str(key).split("/")[-1] == short_name:
            return candidate

    if "distilbert" in model_name.lower() and "distilbert" in profile_lookup:
        return profile_lookup["distilbert"]

    return {}


def summarize_model_profile(checkpoints: dict) -> pd.DataFrame:
    profile_lookup = {
        str(row.get("model")): row
        for row in MODEL_COST_PROFILE
        if isinstance(row, dict) and row.get("model")
    }
    model_order = list(dict.fromkeys([*checkpoints.keys(), *profile_lookup.keys()]))

    rows = []
    for model_name in model_order:
        static_row = _lookup_profile_row(profile_lookup, model_name)
        checkpoint = checkpoints.get(model_name)
        measured = read_checkpoint_metrics(checkpoint)

        row = {
            "model": model_name,
            "checkpoint_available": checkpoint is not None,
            "chunk_f1": measured["chunk_f1"] if measured["chunk_f1"] is not None else static_row.get("chunk_f1"),
            "precision": measured["precision"],
            "recall": measured["recall"],
            "accuracy": measured["accuracy"],
            "train_seconds": measured["train_seconds"]
            if measured["train_seconds"] is not None
            else static_row.get("train_seconds"),
            "params_millions": static_row.get("params_millions"),
            "status": static_row.get("status", "missing_checkpoint"),
        }

        if checkpoint is not None and measured["chunk_f1"] is not None:
            row["status"] = "measured_from_checkpoint"
        elif checkpoint is not None and row["status"] == "missing_checkpoint":
            row["status"] = "checkpoint_available"

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["f1_gain_vs_base"] = np.nan
    df["train_time_ratio_vs_base"] = np.nan

    base_rows = df[(df["model"] == "distilbert") & df["chunk_f1"].notna()]
    if base_rows.empty:
        base_rows = df[df["chunk_f1"].notna()]

    if not base_rows.empty:
        base = base_rows.iloc[0]
        base_f1 = float(base["chunk_f1"])
        df["f1_gain_vs_base"] = pd.to_numeric(df["chunk_f1"], errors="coerce") - base_f1

        base_train = pd.to_numeric(pd.Series([base["train_seconds"]]), errors="coerce").iloc[0]
        if pd.notna(base_train) and base_train > 0:
            df["train_time_ratio_vs_base"] = pd.to_numeric(df["train_seconds"], errors="coerce") / float(base_train)

    return df.sort_values(["checkpoint_available", "chunk_f1"], ascending=[False, False], na_position="last")


def _profile_status_lookup() -> dict:
    return {
        str(row.get("model")): str(row.get("status"))
        for row in MODEL_COST_PROFILE
        if isinstance(row, dict) and row.get("model")
    }


def _profile_status_for_model(status_lookup: dict, model_name: str) -> str:
    if model_name in status_lookup:
        return status_lookup[model_name]

    short_name = model_name.split("/")[-1]
    for key, value in status_lookup.items():
        if str(key).split("/")[-1] == short_name:
            return value

    return "unknown"


project_root = find_project_root(APP_DIR)
checkpoints = available_model_checkpoints(project_root)
available_models = [name for name, checkpoint in checkpoints.items() if checkpoint is not None]

st.title("Shallow Parsing to Event Extraction")
st.write(
    "Interactive dashboard for local CoNLL-2000 chunk checkpoints. "
    "Paste text or an email body and get chunk-aware event records."
)

with st.sidebar:
    st.header("Runtime")
    st.write(f"Project root: {project_root}")

    if not available_models:
        st.error("No model checkpoints found under outputs/. Train or copy checkpoints first.")
        st.stop()

    default_model_idx = available_models.index("distilbert") if "distilbert" in available_models else 0
    model_key = st.selectbox("Model", available_models, index=default_model_idx)
    confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.58, step=0.01)
    mode = st.radio("Input mode", ["Single text", "Email body"], index=0)

pipeline_obj, checkpoint_path = get_pipeline(model_key, str(project_root))
st.caption(f"Loaded checkpoint: {checkpoint_path}")

left, right = st.columns([2, 1])

with right:
    profile_status_lookup = _profile_status_lookup()
    status_rows = []
    for model, checkpoint in checkpoints.items():
        profile_status = _profile_status_for_model(profile_status_lookup, model)
        measured = read_checkpoint_metrics(checkpoint)

        row_status = profile_status
        if checkpoint is not None and measured["chunk_f1"] is not None:
            row_status = "measured_from_checkpoint"

        checkpoint_label = str(checkpoint)
        if checkpoint is None:
            checkpoint_label = "pending_evaluation" if profile_status == "pending_evaluation" else "missing"

        status_rows.append(
            {
                "model": model,
                "checkpoint_available": checkpoint is not None,
                "checkpoint": checkpoint_label,
                "status": row_status,
            }
        )
    st.subheader("Checkpoint Status")
    st.dataframe(streamlit_safe_df(pd.DataFrame(status_rows)), height=260)

    st.subheader("Model Cost Profile")
    profile_df = summarize_model_profile(checkpoints)
    st.dataframe(streamlit_safe_df(profile_df), height=260)

with left:
    if mode == "Single text":
        sample_text = "The company announced a major acquisition in Europe yesterday."
        user_text = st.text_area("Input sentence/text", value=sample_text, height=140)
        run_btn = st.button("Run extraction", type="primary")

        if run_btn:
            record = extract_event_record(user_text, pipeline_obj, confidence_threshold=confidence_threshold)
            summary = {
                key: value
                for key, value in record.items()
                if key not in {"token_predictions", "spans"}
            }

            st.subheader("Structured Output")
            st.dataframe(streamlit_safe_df(pd.DataFrame([summary])), height=120)

            token_df = pd.DataFrame(record["token_predictions"])
            st.subheader("Token-level Chunk Predictions")
            st.dataframe(streamlit_safe_df(token_df), height=280)

            span_df = pd.DataFrame(record["spans"])
            st.subheader("Chunk Spans")
            st.dataframe(streamlit_safe_df(span_df), height=240)

    else:
        sample_email = """Subject: Project Sync and Demo Plan
Hi Team,
Let's meet in Conference Room 2 at 3:30 PM tomorrow to review the release candidate.
Rahul will present the API updates and Sara will demo the dashboard.
Please submit your final notes by Monday evening.
Thanks,
Anita"""
        email_text = st.text_area("Input email body", value=sample_email, height=240)
        run_btn = st.button("Run email pipeline", type="primary")

        if run_btn:
            email_df = run_email_event_pipeline(
                email_text,
                pipeline_obj,
                model_name=model_key,
                confidence_threshold=confidence_threshold,
            )

            if email_df.empty:
                st.warning("No candidate event sentences found.")
            else:
                accepted = int(email_df["accepted"].sum())
                total = int(len(email_df))
                total_cost = float(email_df["cost_estimate"].sum(skipna=True))
                cost_per_accepted = total_cost / accepted if accepted > 0 else float("nan")

                m1, m2, m3 = st.columns(3)
                m1.metric("Accepted events", f"{accepted}/{total}")
                m2.metric("Total cost estimate", f"{total_cost:.2f}")
                m3.metric(
                    "Cost / accepted",
                    "n/a" if np.isnan(cost_per_accepted) else f"{cost_per_accepted:.2f}",
                )

                st.subheader("Extracted Event Records")
                st.dataframe(streamlit_safe_df(email_df), height=320)

                event_counts = email_df["event_type"].astype("object").value_counts()
                st.subheader("Event Type Distribution")

                # Matplotlib avoids legacy Streamlit dataframe marshalling issues on pandas string dtypes.
                import matplotlib.pyplot as plt

                plt.style.use("ggplot")

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(event_counts.index.astype(str).tolist(), event_counts.values.tolist())
                ax.set_xlabel("event_type")
                ax.set_ylabel("count")
                ax.set_title("Event Type Distribution")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.download_button(
                    "Download CSV",
                    data=email_df.to_csv(index=False),
                    file_name="email_event_records.csv",
                    mime="text/csv",
                )
