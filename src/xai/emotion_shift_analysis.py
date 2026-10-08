from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from .faithfulness import _predict_one


def emotion_shift_masking_analysis(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int) -> pd.DataFrame:
    """Disable previous-emotion signal to quantify transition reasoning."""
    base = _predict_one(model, tokenizer, sample, labels, device, max_length, use_shift=True)
    masked = _predict_one(model, tokenizer, sample, labels, device, max_length, use_shift=False)
    previous = sample.get("previous_emotion")
    current = sample.get("label_name")
    row = {
        "dialogue_id": sample.get("dialogue_id"),
        "turn_id": sample.get("turn_id"),
        "speaker": sample.get("speaker"),
        "previous_emotion": previous,
        "previous_speaker": sample.get("previous_speaker"),
        "previous_utterance": sample.get("previous_utterance"),
        "current_emotion": current,
        "transition": f"{previous}->{current}" if previous else f"START->{current}",
        "utterance": sample.get("utterance"),
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


def transition_matrix_from_samples(samples: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in samples:
        prev = s.get("previous_emotion") or "START"
        cur = s.get("label_name")
        if cur is not None:
            rows.append({"previous_emotion": prev, "current_emotion": cur})
    if not rows:
        return pd.DataFrame()
    return pd.crosstab(pd.DataFrame(rows)["previous_emotion"], pd.DataFrame(rows)["current_emotion"])
