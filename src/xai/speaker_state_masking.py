from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from .faithfulness import _predict_one


def speaker_state_masking_analysis(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int) -> pd.DataFrame:
    """Mask same-speaker history to quantify speaker-state contribution."""
    base = _predict_one(model, tokenizer, sample, labels, device, max_length, include_speaker_history=True)
    masked = _predict_one(model, tokenizer, sample, labels, device, max_length, include_speaker_history=False)
    row = {
        "dialogue_id": sample.get("dialogue_id"),
        "turn_id": sample.get("turn_id"),
        "speaker": sample.get("speaker"),
        "gold": sample.get("label_name"),
        "utterance": sample.get("utterance"),
        "speaker_history": sample.get("speaker_history", ""),
        "num_speaker_history_turns": len(sample.get("speaker_history_turns", []) or []),
        "base_pred": base["pred"],
        "masked_pred": masked["pred"],
        "base_confidence": base["confidence"],
        "masked_confidence": masked["confidence"],
        "base_gold_probability": base["gold_probability"],
        "masked_gold_probability": masked["gold_probability"],
        "gold_probability_drop": base["gold_probability"] - masked["gold_probability"],
        "prediction_changed": base["pred"] != masked["pred"],
    }
    return pd.DataFrame([row])
