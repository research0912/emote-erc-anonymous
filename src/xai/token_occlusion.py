import re
from copy import deepcopy
from typing import Dict, List
from src.xai.faithfulness import _predict_one


def simple_token_occlusion(model, tokenizer, sample: Dict, labels: List[str], device: str, max_length: int, max_tokens: int = 25):
    base = _predict_one(model, tokenizer, sample, labels, device, max_length)
    words = re.findall(r"\w+|[^\w\s]", sample["utterance"])
    rows = []
    mask_token = getattr(tokenizer, "mask_token", None) or getattr(tokenizer, "unk_token", "[MASK]")
    for i, word in enumerate(words[:max_tokens]):
        modified = deepcopy(sample)
        masked_words = list(words)
        masked_words[i] = mask_token
        modified["utterance"] = " ".join(masked_words)
        masked = _predict_one(model, tokenizer, modified, labels, device, max_length)
        rows.append({
            "token_index": i,
            "token": word,
            "base_gold_probability": base["gold_probability"],
            "masked_gold_probability": masked["gold_probability"],
            "gold_probability_drop": base["gold_probability"] - masked["gold_probability"],
            "pred_probability_drop": base["confidence"] - masked["probabilities"][base["pred_id"]],
            "base_pred": base["pred"],
            "masked_pred": masked["pred"],
        })
    return rows
