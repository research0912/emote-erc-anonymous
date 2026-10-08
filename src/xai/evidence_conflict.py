import torch
from typing import Dict, Any, List
from collections import Counter
from src.xai.faithfulness import _predict_one

def get_stream_prediction(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int, stream_name: str) -> str:
    """Runs prediction with only the specified stream active (others ablated/masked)."""
    if stream_name == "text":
        # Disables context, speaker state, and emotion shift
        res = _predict_one(
            model, tokenizer, sample, labels, device, max_length,
            include_context=False, include_speaker_history=False, use_shift=False
        )
        return res["pred"]
    elif stream_name == "context":
        # Keeps context, disables speaker state and emotion shift
        res = _predict_one(
            model, tokenizer, sample, labels, device, max_length,
            include_context=True, include_speaker_history=False, use_shift=False
        )
        return res["pred"]
    elif stream_name == "speaker":
        # Keeps speaker state, disables context and emotion shift
        res = _predict_one(
            model, tokenizer, sample, labels, device, max_length,
            include_context=False, include_speaker_history=True, use_shift=False
        )
        return res["pred"]
    elif stream_name == "shift":
        # Keeps emotion shift, disables context and speaker state
        res = _predict_one(
            model, tokenizer, sample, labels, device, max_length,
            include_context=False, include_speaker_history=False, use_shift=True
        )
        return res["pred"]
    else:
        raise ValueError(f"Unknown stream: {stream_name}")

def analyze_evidence_conflict(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int, consensus: str = None) -> Dict[str, Any]:
    """
    Computes stream-specific predicted emotions for Text, Context, Speaker State, and Emotion Shift,
    and returns a Conflict Score indicating evidence contradiction.
    """
    # Find which streams are active in the model configuration
    active_streams = ["text"]
    if model.use_bilstm or model.use_gnn:
        active_streams.append("context")
    if getattr(model, "use_speaker_state", False) or model.cfg.get("model", {}).get("use_speaker_state", False):
        if "speaker" not in active_streams: active_streams.append("speaker")
    if getattr(model, "use_emotion_shift", False) or model.cfg.get("model", {}).get("use_emotion_shift", False):
        if "shift" not in active_streams: active_streams.append("shift")
        
    predictions = {}
    for stream in active_streams:
        predictions[stream] = get_stream_prediction(model, tokenizer, sample, labels, device, max_length, stream)
        
    # Consensus: Majority vote of individual stream predictions
    from collections import Counter
    vote_counts = Counter(predictions.values())
    majority_vote_pred = vote_counts.most_common(1)[0][0]
    
    # Only use passed consensus as a tie-breaker if multiple emotions have same vote count
    highest_votes = vote_counts.most_common(1)[0][1]
    tied_emotions = [emo for emo, count in vote_counts.items() if count == highest_votes]
    if consensus is not None and consensus in tied_emotions:
        consensus_to_use = consensus
    else:
        consensus_to_use = majority_vote_pred
    
    # Compute conflict score: fraction of active streams that disagree with the consensus
    disagreeing = sum(1 for s in active_streams if predictions[s] != consensus_to_use)
    conflict_score = disagreeing / len(active_streams)
    
    return {
        "predictions": predictions,
        "consensus": consensus_to_use,
        "conflict_score": round(conflict_score, 4)
    }
