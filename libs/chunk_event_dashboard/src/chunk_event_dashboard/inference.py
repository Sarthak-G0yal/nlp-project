"""Model loading and token prediction helpers."""

from __future__ import annotations

import json
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


def discover_model_roots(project_root: Path) -> Dict[str, str]:
    """Return configured model roots plus any auto-discovered output folders."""
    model_roots = dict(MODEL_ROOTS)
    known_paths = set(model_roots.values())

    def canonical_key(raw_key: str) -> str:
        if raw_key in model_roots:
            return raw_key

        suffix_matches = [key for key in model_roots if key.split("/")[-1] == raw_key]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        return raw_key

    candidate_outputs_dirs = [
        project_root / "outputs",
        project_root / "ipynb" / "outputs",
    ]

    for outputs_dir in candidate_outputs_dirs:
        if not outputs_dir.exists() or not outputs_dir.is_dir():
            continue

        for path in outputs_dir.iterdir():
            if not path.is_dir():
                continue

            relative = str(path.relative_to(project_root))

            if path.name.startswith("scale-study-"):
                inferred_key = path.name.replace("scale-study-", "", 1)
            else:
                inferred_key = path.name

            model_key = canonical_key(inferred_key)

            existing_relative = model_roots.get(model_key)
            if existing_relative is None:
                if relative in known_paths:
                    # Path is already represented by a configured key.
                    continue
                model_roots[model_key] = relative
                known_paths.add(relative)
                continue

            if existing_relative == relative or relative in known_paths:
                continue

            existing_checkpoint = pick_latest_checkpoint(project_root / existing_relative)
            candidate_checkpoint = pick_latest_checkpoint(project_root / relative)

            # Keep the configured path unless the alternative has a checkpoint and the configured one does not.
            if existing_checkpoint is None and candidate_checkpoint is not None:
                model_roots[model_key] = relative
                known_paths.add(relative)

    return model_roots


def resolve_model_checkpoint(
    model_key: str,
    project_root: Path,
    model_roots: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    roots = model_roots or MODEL_ROOTS
    relative = roots.get(model_key)
    if relative is None:
        raise KeyError(f"Unknown model key: {model_key}")
    return pick_latest_checkpoint(project_root / relative)


def available_model_checkpoints(project_root: Path) -> Dict[str, Optional[Path]]:
    model_roots = discover_model_roots(project_root)
    return {
        model_key: resolve_model_checkpoint(model_key, project_root, model_roots=model_roots)
        for model_key in model_roots
    }


def _to_optional_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_checkpoint_metrics(checkpoint: Optional[Path]) -> Dict[str, Optional[float]]:
    """Read eval metrics from a checkpoint trainer_state.json when available."""
    metrics = {
        "chunk_f1": None,
        "precision": None,
        "recall": None,
        "accuracy": None,
        "train_seconds": None,
    }
    if checkpoint is None:
        return metrics

    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        return metrics

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return metrics

    metrics["chunk_f1"] = _to_optional_float(state.get("best_metric"))
    log_history = state.get("log_history")
    if isinstance(log_history, list):
        latest_eval = next(
            (entry for entry in reversed(log_history) if isinstance(entry, dict) and "eval_f1" in entry),
            None,
        )
        if latest_eval is not None:
            metrics["precision"] = _to_optional_float(latest_eval.get("eval_precision"))
            metrics["recall"] = _to_optional_float(latest_eval.get("eval_recall"))
            metrics["accuracy"] = _to_optional_float(latest_eval.get("eval_accuracy"))
            if metrics["chunk_f1"] is None:
                metrics["chunk_f1"] = _to_optional_float(latest_eval.get("eval_f1"))

        latest_train = next(
            (entry for entry in reversed(log_history) if isinstance(entry, dict) and "train_runtime" in entry),
            None,
        )
        if latest_train is not None:
            metrics["train_seconds"] = _to_optional_float(latest_train.get("train_runtime"))

    return metrics


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
    model_roots = discover_model_roots(project_root)
    checkpoint = resolve_model_checkpoint(model_key, project_root, model_roots=model_roots)
    if checkpoint is None:
        expected_root = model_roots.get(model_key, "unknown")
        raise FileNotFoundError(
            f"No checkpoint found for {model_key}. Expected under: {project_root / expected_root}"
        )

    from transformers import pipeline

    pipe = pipeline(
        "token-classification",
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        aggregation_strategy="none",
    )
    return pipe, checkpoint
