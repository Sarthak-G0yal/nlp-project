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
from chunk_event_dashboard.inference import available_model_checkpoints, find_project_root, load_token_classifier


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


def summarize_model_profile() -> pd.DataFrame:
    df = pd.DataFrame(MODEL_COST_PROFILE)
    if df.empty:
        return df
    base = df[df["model"] == "distilbert"].iloc[0]
    df["f1_gain_vs_base"] = df["chunk_f1"] - float(base["chunk_f1"])
    df["train_time_ratio_vs_base"] = df["train_seconds"] / float(base["train_seconds"])
    return df


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
    status_rows = [
        {
            "model": model,
            "checkpoint_available": checkpoints[model] is not None,
            "checkpoint": str(checkpoints[model]) if checkpoints[model] else "missing",
        }
        for model in checkpoints
    ]
    st.subheader("Checkpoint Status")
    st.dataframe(streamlit_safe_df(pd.DataFrame(status_rows)), height=260)

    st.subheader("Model Cost Profile")
    profile_df = summarize_model_profile()
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
