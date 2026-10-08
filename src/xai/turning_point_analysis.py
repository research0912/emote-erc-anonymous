import pandas as pd
from typing import Dict, List, Any
from src.xai.context_masking import context_masking_analysis

def analyze_turning_points(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int, precomputed_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
    """
    Identifies conversational turning points by measuring the drop in gold emotion probability
    when each context turn is removed.
    """
    if precomputed_df is not None:
        df = precomputed_df
    else:
        _, df = context_masking_analysis(model, tokenizer, sample, labels, device, max_length)
    if df.empty:
        return []
        
    turning_points = []
    for _, row in df.iterrows():
        # High probability drop = high turning point score/importance
        importance = float(row.get("pred_probability_drop", row.get("gold_probability_drop", 0.0)))
        turning_points.append({
            "turn_id": int(row["masked_turn_id"]),
            "context_turn": f"{row['masked_speaker']}: {row['masked_utterance']}",
            "importance": max(0.0, importance), # Clamp negative values to 0.0
            "prediction_changed": bool(row["prediction_changed"]),
            "base_pred": row["base_pred"],
            "masked_pred": row["masked_pred"]
        })
        
    # Sort by importance descending
    turning_points = sorted(turning_points, key=lambda x: x["importance"], reverse=True)
    return turning_points
