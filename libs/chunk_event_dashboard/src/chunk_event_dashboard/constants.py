"""Shared constants aligned with notebook implementations."""

from __future__ import annotations

import re

CHUNK_LABELS = [
    "O",
    "B-ADJP",
    "I-ADJP",
    "B-ADVP",
    "I-ADVP",
    "B-CONJP",
    "I-CONJP",
    "B-INTJ",
    "I-INTJ",
    "B-LST",
    "I-LST",
    "B-NP",
    "I-NP",
    "B-PP",
    "I-PP",
    "B-PRT",
    "I-PRT",
    "B-SBAR",
    "I-SBAR",
    "B-UCP",
    "I-UCP",
    "B-VP",
    "I-VP",
]

MODEL_ROOTS = {
    "distilbert": "outputs/distilbert-conll2000",
    "bert-base-uncased": "outputs/scale-study-bert-base-uncased",
    "roberta-base": "outputs/scale-study-roberta-base",
    "google/bert_uncased_L-2_H-128_A-2": "outputs/scale-study-bert_uncased_L-2_H-128_A-2",
    "prajjwal1/bert-tiny": "outputs/scale-study-bert-tiny",
    "huawei-noah/TinyBERT_General_4L_312D": "outputs/scale-study-TinyBERT_General_4L_312D",
}

EVENT_KEYWORDS = {
    "meeting": ["meet", "meeting", "sync", "call", "discussion", "demo"],
    "deadline": ["deadline", "due", "submit", "delivery", "by"],
    "travel": ["travel", "flight", "hotel", "trip", "visit"],
    "incident": ["incident", "issue", "error", "outage", "failure"],
    "announcement": ["announce", "announcement", "release", "launched"],
}

ROLE_REQUIRED = {
    "meeting": ["trigger", "location_or_time"],
    "deadline": ["trigger", "time"],
    "travel": ["trigger", "location"],
    "incident": ["trigger"],
    "announcement": ["trigger", "subject"],
    "other": ["trigger"],
}

TIME_PATTERN = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|tonight|\d{1,2}[:.]\d{2}\s?(am|pm)?|\d{1,2}\s?(am|pm)|\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?)\b",
    re.IGNORECASE,
)

LOCATION_PATTERN = re.compile(
    r"\b(in|at|from|to|near|inside|within|across|around|into|onto)\s+([A-Za-z0-9][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9\-]*){0,6})",
    re.IGNORECASE,
)

MODEL_COST_PROFILE = [
    {
        "model": "distilbert",
        "chunk_f1": 0.9566,
        "train_seconds": 120.0704,
        "params_millions": 66.3806,
        "status": "measured",
    },
    {
        "model": "bert-base-uncased",
        "chunk_f1": 0.9602,
        "train_seconds": 222.7203,
        "params_millions": 108.9093,
        "status": "measured",
    },
    {
        "model": "roberta-base",
        "chunk_f1": 0.9665,
        "train_seconds": 224.6860,
        "params_millions": 124.0727,
        "status": "measured",
    },
    {
        "model": "google/bert_uncased_L-2_H-128_A-2",
        "chunk_f1": None,
        "train_seconds": None,
        "params_millions": None,
        "status": "pending_evaluation",
    },
    {
        "model": "prajjwal1/bert-tiny",
        "chunk_f1": None,
        "train_seconds": None,
        "params_millions": None,
        "status": "pending_evaluation",
    },
    {
        "model": "huawei-noah/TinyBERT_General_4L_312D",
        "chunk_f1": None,
        "train_seconds": None,
        "params_millions": None,
        "status": "pending_evaluation",
    },
]
