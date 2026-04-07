"""Model loading and token prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import CHUNK_LABELS, MODEL_ROOTS


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the repository root by walking up until pyproject and outputs exist."""
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent

    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "outputs").exists():
            return parent

    raise FileNotFoundError("Could not locate project root with pyproject.toml and outputs/")


def pick_latest_checkpoint(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None

    checkpoints = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-") and (path / "config.json").exists()
    ]

    if checkpoints:
        return sorted(checkpoints, key=lambda p: int(p.name.split("-")[-1]))[-1]

    if (base_dir / "config.json").exists():
        return base_dir

    return None


def resolve_model_checkpoint(model_key: str, project_root: Path) -> Optional[Path]:
    relative = MODEL_ROOTS.get(model_key)
    if relative is None:
        raise KeyError(f"Unknown model key: {model_key}")
    return pick_latest_checkpoint(project_root / relative)


def available_model_checkpoints(project_root: Path) -> Dict[str, Optional[Path]]:
    return {model_key: resolve_model_checkpoint(model_key, project_root) for model_key in MODEL_ROOTS}


def normalize_label(tag: str) -> str:
    if not isinstance(tag, str) or not tag:
        return "O"
    normalized = tag.replace("LABEL_", "")
    if normalized.isdigit():
        idx = int(normalized)
        if 0 <= idx < len(CHUNK_LABELS):
            return CHUNK_LABELS[idx]
    return normalized


def merge_subword_predictions(pred_items: List[dict]) -> Tuple[List[str], List[str], List[float]]:
    merged_tokens: List[str] = []
    merged_labels: List[str] = []
    merged_scores: List[float] = []

    for item in pred_items:
        token = str(item.get("word", ""))
        label = normalize_label(str(item.get("entity") or item.get("entity_group") or "O"))
        score = float(item.get("score", 0.5))

        is_subword = token.startswith("##")
        token_clean = token[2:] if is_subword else token
        if token_clean.startswith("\u0120") or token_clean.startswith("\u2581"):
            token_clean = token_clean[1:]

        if is_subword and merged_tokens:
            merged_tokens[-1] += token_clean
            merged_scores[-1] = min(merged_scores[-1], score)
            continue

        merged_tokens.append(token_clean)
        merged_labels.append(label)
        merged_scores.append(score)

    n = min(len(merged_tokens), len(merged_labels), len(merged_scores))
    return merged_tokens[:n], merged_labels[:n], merged_scores[:n]


def load_token_classifier(model_key: str, project_root: Path):
    """Load a Hugging Face token-classification pipeline from local checkpoints."""
    checkpoint = resolve_model_checkpoint(model_key, project_root)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found for {model_key}. Expected under: {project_root / MODEL_ROOTS[model_key]}"
        )

    from transformers import pipeline

    pipe = pipeline(
        "token-classification",
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        aggregation_strategy="none",
    )
    return pipe, checkpoint
