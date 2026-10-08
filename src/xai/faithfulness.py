from copy import deepcopy
from typing import Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase
from src.data_loader.base import build_model_text, build_speaker_state_text


def _current_text(sample):
    return f"{sample.get('speaker', 'speaker')}: {sample.get('utterance', '')}"


def _turn_texts(sample, include_context=True):
    turns = sample.get("context_turns", []) if include_context else []
    texts = [f"{turn.get('speaker', 'speaker')}: {turn.get('utterance', '')}" for turn in turns]
    texts.append(_current_text(sample))
    return texts


def _turn_inputs(model, tokenizer, sample, device, include_context=True):
    turn_max_length = model.cfg.get("dataset", {}).get("turn_max_length", 96)
    placeholder = tokenizer.unk_token or tokenizer.pad_token or "[NO_TEXT]"
    turns = _turn_texts(sample, include_context=include_context)
    if not turns:
        turns = [placeholder]
    enc = tokenizer(
        turns,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=turn_max_length,
    )
    num_turns = len(turns)
    return {
        "context_input_ids": enc["input_ids"].unsqueeze(0).to(device),
        "context_attention_mask": enc["attention_mask"].unsqueeze(0).to(device),
        "context_turn_mask": torch.ones(1, num_turns, dtype=torch.float32, device=device),
        "current_turn_indices": torch.tensor([num_turns - 1], dtype=torch.long, device=device),
    }


def _predict_one(model, tokenizer: PreTrainedTokenizerBase, sample: Dict, labels: List[str], device: str,
                 max_length: int, include_context=True, include_speaker_history=True, use_shift=True,
                 ablation: Optional[str] = None):
    # Apply input-level ablations/masking to double-guarantee zero leakage
    if ablation == "text":
        sample = deepcopy(sample)
        mask_token = getattr(tokenizer, "mask_token", None) or getattr(tokenizer, "unk_token", "[MASK]")
        sample["utterance"] = mask_token
        
    if ablation == "dialogue_flow":
        include_context = False
        
    if ablation == "speaker_state":
        include_speaker_history = False
        
    if ablation == "emotion_shift":
        use_shift = False

    use_turn_level_flow = model.cfg.get("model", {}).get("dialogue_flow_source") == "turn_level"
    include_main_context = model.cfg.get("model", {}).get("main_input_include_context_when_turn_level", False)
    text = (
        _current_text(sample)
        if use_turn_level_flow and not include_main_context
        else build_model_text(sample, include_context=include_context)
    )
    spk_text = build_speaker_state_text(sample, include_speaker_history=include_speaker_history)
    has_previous_turn = bool(sample.get("previous_turn_text"))
    previous_text = sample.get("previous_turn_text") or tokenizer.unk_token or "[NO_PREVIOUS_TURN]"
    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    spk = tokenizer([spk_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    prev_enc = tokenizer([previous_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    prev = torch.tensor([sample.get("previous_emotion_id", -100)], dtype=torch.long, device=device)
    previous_turn_exists = torch.tensor([1.0 if has_previous_turn else 0.0], dtype=torch.float32, device=device)
    
    if ablation is None and not use_shift:
        ablation = "no_shift"
        
    turn_inputs = _turn_inputs(model, tokenizer, sample, device, include_context=include_context) if use_turn_level_flow else {}
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            speaker_input_ids=spk["input_ids"], speaker_attention_mask=spk["attention_mask"],
            previous_input_ids=prev_enc["input_ids"], previous_attention_mask=prev_enc["attention_mask"],
            previous_turn_exists=previous_turn_exists,
            previous_emotion_ids=prev, ablation=ablation,
            **turn_inputs,
        )["logits"]
        probs = logits.softmax(-1)[0]
    pred_id = int(probs.argmax())
    gold_id = int(sample["label"])
    return {
        "pred_id": pred_id,
        "pred": labels[pred_id],
        "confidence": float(probs[pred_id]),
        "gold_probability": float(probs[gold_id]),
        "probabilities": probs.detach().cpu().tolist(),
    }


def context_turn_importance(model, tokenizer, sample, labels, device, max_length):
    base = _predict_one(model, tokenizer, sample, labels, device, max_length)
    rows = []
    turns = sample.get("context_turns", [])
    for idx, turn in enumerate(turns):
        modified = deepcopy(sample)
        kept = [t for j, t in enumerate(turns) if j != idx]
        modified["context_turns"] = kept
        modified["context"] = " </s> ".join([f"{t['speaker']}: {t['utterance']}" for t in kept])
        masked = _predict_one(model, tokenizer, modified, labels, device, max_length)
        rows.append({
            "masked_context_index": idx,
            "masked_turn_id": turn["turn_id"],
            "masked_speaker": turn["speaker"],
            "masked_utterance": turn["utterance"],
            "base_pred": base["pred"],
            "masked_pred": masked["pred"],
            "base_gold_probability": base["gold_probability"],
            "masked_gold_probability": masked["gold_probability"],
            "gold_probability_drop": base["gold_probability"] - masked["gold_probability"],
            "pred_probability_drop": base["confidence"] - masked["probabilities"][base["pred_id"]],
        })
    return base, rows


def speaker_state_importance(model, tokenizer, sample, labels, device, max_length):
    base = _predict_one(model, tokenizer, sample, labels, device, max_length, include_speaker_history=True)
    masked = _predict_one(model, tokenizer, sample, labels, device, max_length, include_speaker_history=False)
    return {
        "base_pred": base["pred"],
        "masked_pred": masked["pred"],
        "base_gold_probability": base["gold_probability"],
        "masked_gold_probability": masked["gold_probability"],
        "gold_probability_drop": base["gold_probability"] - masked["gold_probability"],
        "pred_probability_drop": base["confidence"] - masked["probabilities"][base["pred_id"]],
    }


def shift_importance(model, tokenizer, sample, labels, device, max_length):
    base = _predict_one(model, tokenizer, sample, labels, device, max_length, use_shift=True)
    masked = _predict_one(model, tokenizer, sample, labels, device, max_length, use_shift=False)
    return {
        "previous_emotion": sample.get("previous_emotion"),
        "previous_speaker": sample.get("previous_speaker"),
        "previous_utterance": sample.get("previous_utterance"),
        "base_pred": base["pred"],
        "masked_pred": masked["pred"],
        "base_gold_probability": base["gold_probability"],
        "masked_gold_probability": masked["gold_probability"],
        "gold_probability_drop": base["gold_probability"] - masked["gold_probability"],
        "pred_probability_drop": base["confidence"] - masked["probabilities"][base["pred_id"]],
    }
