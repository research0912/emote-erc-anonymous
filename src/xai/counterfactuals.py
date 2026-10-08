from copy import deepcopy
from typing import Dict, Any, List
from src.xai.faithfulness import _predict_one
from src.xai.context_masking import context_masking_analysis
from src.xai.speaker_state_masking import speaker_state_masking_analysis
from src.xai.emotion_shift_analysis import emotion_shift_masking_analysis

def compute_counterfactual(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int, precomputed_ctx_df: Any = None) -> Dict[str, Any]:
    """
    Finds the minimal perturbation that changes the model's predicted label.
    Checks, in prioritized order:
    1. Removing individual context turns
    2. Disabling speaker state (same-speaker history)
    3. Disabling previous emotion shift
    4. Removing all context turns
    """
    base = _predict_one(model, tokenizer, sample, labels, device, max_length)
    original_pred = base["pred"]
    
    # 1. Test individual context turns removal
    if precomputed_ctx_df is not None:
        ctx_df = precomputed_ctx_df
    else:
        _, ctx_df = context_masking_analysis(model, tokenizer, sample, labels, device, max_length)
    if not ctx_df.empty:
        # Sort by drop in gold probability, or just check the first one that changed prediction
        changed_turns = ctx_df[ctx_df["prediction_changed"]]
        if not changed_turns.empty:
            # Get the turn that caused prediction change with highest drop
            best_turn = changed_turns.sort_values("gold_probability_drop", ascending=False).iloc[0]
            return {
                "original": original_pred,
                "counterfactual": best_turn["masked_pred"],
                "required_change": f"remove context turn {best_turn['masked_turn_id']} ('{best_turn['masked_speaker']}: {best_turn['masked_utterance']}')"
            }
            
    # 2. Test disabling speaker state
    spk_df = speaker_state_masking_analysis(model, tokenizer, sample, labels, device, max_length)
    if not spk_df.empty and spk_df.iloc[0]["prediction_changed"]:
        return {
            "original": original_pred,
            "counterfactual": spk_df.iloc[0]["masked_pred"],
            "required_change": "remove speaker state (same-speaker history)"
        }
        
    # 3. Test disabling emotion shift
    shf_df = emotion_shift_masking_analysis(model, tokenizer, sample, labels, device, max_length)
    if not shf_df.empty and shf_df.iloc[0]["prediction_changed"]:
        return {
            "original": original_pred,
            "counterfactual": shf_df.iloc[0]["masked_pred"],
            "required_change": "remove previous emotion shift context"
        }
        
    # 4. Test removing all context turns
    modified_no_ctx = deepcopy(sample)
    modified_no_ctx["context_turns"] = []
    modified_no_ctx["context"] = ""
    no_ctx_pred = _predict_one(model, tokenizer, modified_no_ctx, labels, device, max_length)
    if no_ctx_pred["pred"] != original_pred:
        return {
            "original": original_pred,
            "counterfactual": no_ctx_pred["pred"],
            "required_change": "remove all context turns"
        }
        
    # If no single change modified the prediction
    return {
        "original": original_pred,
        "counterfactual": original_pred,
        "required_change": "no single stream modification changes the prediction (highly robust)"
    }
