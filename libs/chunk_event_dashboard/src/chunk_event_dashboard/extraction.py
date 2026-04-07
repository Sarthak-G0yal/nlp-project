"""Chunk-to-event extraction heuristics reused from notebooks."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import EVENT_KEYWORDS, LOCATION_PATTERN, MODEL_COST_PROFILE, ROLE_REQUIRED, TIME_PATTERN
from .inference import merge_subword_predictions


def bio_to_spans(tokens: List[str], chunk_labels: List[str]) -> List[Dict]:
    spans = []
    cur_type = None
    cur_tokens: List[str] = []
    cur_start = None

    for i, (tok, tag) in enumerate(zip(tokens, chunk_labels)):
        if tag == "O":
            if cur_type is not None:
                spans.append({"type": cur_type, "text": " ".join(cur_tokens), "start": cur_start, "end": i - 1})
                cur_type, cur_tokens, cur_start = None, [], None
            continue

        prefix, ctype = tag.split("-", 1) if "-" in tag else ("B", tag)
        if prefix == "B" or ctype != cur_type:
            if cur_type is not None:
                spans.append({"type": cur_type, "text": " ".join(cur_tokens), "start": cur_start, "end": i - 1})
            cur_type, cur_tokens, cur_start = ctype, [tok], i
        else:
            cur_tokens.append(tok)

    if cur_type is not None:
        spans.append(
            {
                "type": cur_type,
                "text": " ".join(cur_tokens),
                "start": cur_start,
                "end": len(tokens) - 1,
            }
        )

    return spans


def sentence_to_ie_record(tokens: List[str], chunk_labels: List[str]) -> Dict:
    spans = bio_to_spans(tokens, chunk_labels)
    nps = [s for s in spans if s["type"] == "NP"]
    vps = [s for s in spans if s["type"] == "VP"]

    subject = nps[0]["text"] if nps else None
    action = vps[0]["text"] if vps else None

    obj = None
    if vps:
        vp_end = vps[0]["end"]
        for np_span in nps:
            if np_span["start"] > vp_end:
                obj = np_span["text"]
                break

    return {
        "subject": subject,
        "action": action,
        "object": obj,
        "num_chunks": len(spans),
        "spans": spans,
    }


def classify_event_type(trigger: Optional[str], sentence: str) -> str:
    text = f"{trigger or ''} {sentence or ''}".lower()
    for event_type, keys in EVENT_KEYWORDS.items():
        if any(key in text for key in keys):
            return event_type
    return "other"


def extract_location_phrases(sentence: str) -> List[str]:
    found = []
    for match in LOCATION_PATTERN.finditer(sentence or ""):
        phrase = f"{match.group(1)} {match.group(2)}".strip()
        if TIME_PATTERN.search(phrase):
            continue
        found.append(phrase)
    return list(dict.fromkeys([item.lower() for item in found]))


def extract_time_phrases(sentence: str) -> List[str]:
    found = [match.group(0).strip() for match in TIME_PATTERN.finditer(sentence or "")]
    return list(dict.fromkeys([item.lower() for item in found]))


def role_completeness_score(
    event_type: str,
    subject: Optional[str],
    trigger: Optional[str],
    obj: Optional[str],
    locations: List[str],
    times: List[str],
) -> float:
    has_location_or_time = bool(locations or times)
    provided = {
        "trigger": bool(trigger),
        "subject": bool(subject),
        "object": bool(obj),
        "location": bool(locations),
        "time": bool(times),
        "location_or_time": has_location_or_time,
    }
    required = ROLE_REQUIRED.get(event_type, ["trigger"])
    ok = sum(1 for key in required if provided.get(key, False))
    return ok / max(len(required), 1)


def extract_event_record(sentence: str, pipeline_obj, confidence_threshold: float = 0.55) -> Dict:
    pred = pipeline_obj(sentence)
    tokens, labels, scores = merge_subword_predictions(pred)

    if not tokens:
        return {
            "sentence": sentence,
            "event_type": "other",
            "trigger": None,
            "subject": None,
            "object": None,
            "location": None,
            "time": None,
            "confidence": 0.0,
            "abstained": True,
            "abstain_reason": "no_tokens",
            "num_chunks": 0,
            "tokens_processed": 0,
            "spans": [],
            "token_predictions": [],
        }

    ie = sentence_to_ie_record(tokens, labels)
    trigger = ie["action"]
    event_type = classify_event_type(trigger, sentence)

    locations = extract_location_phrases(sentence)
    times = extract_time_phrases(sentence)

    mean_token_conf = float(np.mean(scores)) if scores else 0.5
    role_score = role_completeness_score(event_type, ie["subject"], trigger, ie["object"], locations, times)
    lexical_bonus = 1.0 if event_type != "other" else 0.7
    confidence = float(np.clip(0.55 * mean_token_conf + 0.35 * role_score + 0.10 * lexical_bonus, 0.0, 1.0))

    abstained = confidence < confidence_threshold
    abstain_reason = "low_confidence" if abstained else None

    token_predictions = [
        {"token": token, "chunk_label": label, "score": score}
        for token, label, score in zip(tokens, labels, scores)
    ]

    return {
        "sentence": sentence,
        "event_type": event_type,
        "trigger": trigger,
        "subject": ie["subject"],
        "object": ie["object"],
        "location": "; ".join(locations) if locations else None,
        "time": "; ".join(times) if times else None,
        "confidence": confidence,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "num_chunks": ie["num_chunks"],
        "tokens_processed": len(tokens),
        "spans": ie["spans"],
        "token_predictions": token_predictions,
    }


def split_email_into_sentences(email_text: str) -> List[str]:
    lines = [ln.strip() for ln in (email_text or "").replace("\r", "").split("\n") if ln.strip()]
    lines = [ln for ln in lines if not re.match(r"^(subject|from|to|cc|bcc):", ln, flags=re.IGNORECASE)]
    sentences = []
    for line in lines:
        sentences.extend([part.strip() for part in re.split(r"(?<=[.!?])\s+", line) if part.strip()])
    return sentences


def estimate_event_cost(tokens_processed: int, model_name: str) -> float:
    base_row = next((row for row in MODEL_COST_PROFILE if row["model"] == "distilbert"), None)
    model_row = next((row for row in MODEL_COST_PROFILE if row["model"] == model_name), None)
    if not base_row or not model_row:
        return float("nan")

    base_params = base_row.get("params_millions")
    model_params = model_row.get("params_millions")
    if base_params is None or model_params is None:
        return float("nan")

    params_scale = float(model_params) / float(base_params)
    return float(tokens_processed * params_scale)


def run_email_event_pipeline(
    email_text: str,
    pipeline_obj,
    model_name: str = "distilbert",
    confidence_threshold: float = 0.55,
) -> pd.DataFrame:
    rows = []
    for sentence in split_email_into_sentences(email_text):
        rec = extract_event_record(sentence, pipeline_obj, confidence_threshold=confidence_threshold)
        rows.append(
            {
                "sentence": rec["sentence"],
                "event_type": rec["event_type"],
                "trigger": rec["trigger"],
                "subject": rec["subject"],
                "object": rec["object"],
                "location": rec["location"],
                "time": rec["time"],
                "confidence": rec["confidence"],
                "abstained": rec["abstained"],
                "abstain_reason": rec["abstain_reason"],
                "num_chunks": rec["num_chunks"],
                "tokens_processed": rec["tokens_processed"],
                "model": model_name,
                "cost_estimate": estimate_event_cost(rec["tokens_processed"], model_name),
            }
        )

    output = pd.DataFrame(rows)
    if not output.empty:
        output["accepted"] = ~output["abstained"]
    return output
