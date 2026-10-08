import torch
from typing import Dict, Any, List, Tuple
from copy import deepcopy
from src.xai.faithfulness import _predict_one

def compute_stream_attribution(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Computes causal contribution of each active stream using feature-level probability drops:
    Contribution_i = P(original_class) - P(original_class | ablated_stream_i)
    And normalizes so they sum to 1.0.
    """
    contributions = {}
    
    # Base prediction
    base_res = _predict_one(model, tokenizer, sample, labels, device, max_length)
    pred_id = base_res["pred_id"]
    base_prob = base_res["probabilities"][pred_id]
    
    def get_ablated_prob(ablation: str):
        res = _predict_one(model, tokenizer, sample, labels, device, max_length, ablation=ablation)
        return res["probabilities"][pred_id]
        
    # 1. Text Encoder (Contextual representation)
    text_prob = get_ablated_prob("text")
    contributions["text"] = max(0.0, base_prob - text_prob)
    
    # 2. Dialogue Flow (Context)
    is_first_turn = sample.get("turn_id", 0) == 0
    if is_first_turn:
        # Turn 0 has strictly 0 contribution for dialogue flow as there is no context
        contributions["dialogue_flow"] = 0.0
    else:
        flow_prob = get_ablated_prob("dialogue_flow")
        contributions["dialogue_flow"] = max(0.0, base_prob - flow_prob)
        
    # 3. Speaker State
    is_first_turn = sample.get("turn_id", 0) == 0
    if is_first_turn:
        # Turn 0 has strictly 0 contribution for speaker state as there is no speaker history
        contributions["speaker_state"] = 0.0
    else:
        spk_prob = get_ablated_prob("speaker_state")
        contributions["speaker_state"] = max(0.0, base_prob - spk_prob)
        
    # 4. Emotion Shift
    has_previous = bool(sample.get("previous_turn_text")) and (sample.get("turn_id", 0) > 0)
    if has_previous:
        shift_prob = get_ablated_prob("emotion_shift")
        contributions["emotion_shift"] = max(0.0, base_prob - shift_prob)
    else:
        # Turn 0 or no previous turn has strictly 0 contribution for shift
        contributions["emotion_shift"] = 0.0
        
    raw_drops = contributions.copy()

    # Normalize contributions so they sum to 1.0
    total = sum(contributions.values())
    if total >= 0.02: # Avoid normalizing tiny noise drops (under 2% confidence drop)
        for k in contributions:
            contributions[k] /= total
    else:
        # Uniform fallback when confidence changes are negligible
        for k in contributions:
            contributions[k] = 1.0 / len(contributions)
            
    return contributions, raw_drops
