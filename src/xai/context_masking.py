from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple
import pandas as pd
from .faithfulness import _predict_one


def context_masking_analysis(
    model,
    tokenizer,
    sample: Dict[str, Any],
    labels: List[str],
    device: str,
    max_length: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Leave-one-context-turn-out analysis.

    Outcome representation for paper:
    - base prediction and confidence
    - gold probability before/after each context turn is removed
    - confidence drop per context turn
    """
    base = _predict_one(model, tokenizer, sample, labels, device, max_length)
    rows = []
    turns = sample.get("context_turns", []) or []
    for idx, turn in enumerate(turns):
        masked = deepcopy(sample)
        kept = [t for j, t in enumerate(turns) if j != idx]
        masked["context_turns"] = kept
        masked["context"] = " </s> ".join([f"{t.get('speaker', '')}: {t.get('utterance', '')}" for t in kept])
        masked_pred = _predict_one(model, tokenizer, masked, labels, device, max_length)
        rows.append({
            "dialogue_id": sample.get("dialogue_id"),
            "turn_id": sample.get("turn_id"),
            "gold": sample.get("label_name"),
            "utterance": sample.get("utterance"),
            "masked_context_index": idx,
            "masked_turn_id": turn.get("turn_id"),
            "masked_speaker": turn.get("speaker"),
            "masked_utterance": turn.get("utterance"),
            "base_pred": base["pred"],
            "masked_pred": masked_pred["pred"],
            "base_confidence": base["confidence"],
            "masked_confidence": masked_pred["confidence"],
            "base_gold_probability": base["gold_probability"],
            "masked_gold_probability": masked_pred["gold_probability"],
            "gold_probability_drop": base["gold_probability"] - masked_pred["gold_probability"],
            "prediction_changed": base["pred"] != masked_pred["pred"],
        })
    return base, pd.DataFrame(rows)


def summarize_context_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df.sort_values("gold_probability_drop", ascending=False).reset_index(drop=True)
