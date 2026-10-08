from __future__ import annotations

from typing import Dict, List

import torch

from src.data_loader.base import build_model_text, build_speaker_state_text
from src.xai.faithfulness import _current_text, _turn_inputs


def _main_text_for_model(model, sample: Dict, include_context: bool = True) -> str:
    use_turn_level_flow = model.cfg.get("model", {}).get("dialogue_flow_source") == "turn_level"
    include_main_context = model.cfg.get("model", {}).get("main_input_include_context_when_turn_level", False)
    if use_turn_level_flow and not include_main_context:
        return _current_text(sample)
    return build_model_text(sample, include_context=include_context)


def integrated_gradients_main_input(
    model,
    tokenizer,
    sample: Dict,
    labels: List[str],
    device: str,
    max_length: int,
    steps: int = 24,
    target: str = "pred",
):
    """Token-level Integrated Gradients for the model's main text input.

    The model has several conversational streams. This attribution explains the
    main transformer stream, which is the current utterance for older turn-level
    configs and context+current text for optimized configs.
    """
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    text = _main_text_for_model(model, sample, include_context=True)
    spk_text = build_speaker_state_text(sample, include_speaker_history=True)
    previous_text = sample.get("previous_turn_text") or tokenizer.unk_token or "[NO_PREVIOUS_TURN]"
    has_previous_turn = bool(sample.get("previous_turn_text"))

    enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    spk = tokenizer([spk_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    prev_enc = tokenizer([previous_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    previous_turn_exists = torch.tensor([1.0 if has_previous_turn else 0.0], dtype=torch.float32, device=device)
    prev = torch.tensor([sample.get("previous_emotion_id", -100)], dtype=torch.long, device=device)

    use_turn_level_flow = model.cfg.get("model", {}).get("dialogue_flow_source") == "turn_level"
    turn_inputs = _turn_inputs(model, tokenizer, sample, device, include_context=True) if use_turn_level_flow else {}

    embedding_layer = model.encoder.get_input_embeddings()
    input_embeds = embedding_layer(enc["input_ids"]).detach()
    baseline_id = tokenizer.pad_token_id
    if baseline_id is None:
        baseline_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    baseline_ids = torch.full_like(enc["input_ids"], baseline_id)
    baseline_embeds = embedding_layer(baseline_ids).detach()

    with torch.no_grad():
        base_logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            speaker_input_ids=spk["input_ids"],
            speaker_attention_mask=spk["attention_mask"],
            previous_input_ids=prev_enc["input_ids"],
            previous_attention_mask=prev_enc["attention_mask"],
            previous_turn_exists=previous_turn_exists,
            previous_emotion_ids=prev,
            **turn_inputs,
        )["logits"]
        probs = base_logits.softmax(dim=-1)[0]

    pred_id = int(probs.argmax())
    gold_id = int(sample["label"])
    target_id = gold_id if target == "gold" else pred_id

    total_grads = torch.zeros_like(input_embeds)
    for alpha in torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]:
        scaled_embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        scaled_embeds = scaled_embeds.detach().requires_grad_(True)
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            inputs_embeds=scaled_embeds,
            speaker_input_ids=spk["input_ids"],
            speaker_attention_mask=spk["attention_mask"],
            previous_input_ids=prev_enc["input_ids"],
            previous_attention_mask=prev_enc["attention_mask"],
            previous_turn_exists=previous_turn_exists,
            previous_emotion_ids=prev,
            **turn_inputs,
        )["logits"]
        score = logits[0, target_id]
        grads = torch.autograd.grad(score, scaled_embeds, retain_graph=False)[0]
        total_grads += grads.detach()

    avg_grads = total_grads / float(steps)
    attributions = ((input_embeds - baseline_embeds) * avg_grads).sum(dim=-1)[0]
    token_ids = enc["input_ids"][0].detach().cpu().tolist()
    attention_mask = enc["attention_mask"][0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    special_ids = set(tokenizer.all_special_ids)

    rows = []
    for idx, (token_id, mask_value, token, score) in enumerate(zip(token_ids, attention_mask, tokens, attributions.detach().cpu().tolist())):
        if not mask_value or token_id in special_ids:
            continue
        rows.append({
            "token_index": idx,
            "token": token,
            "attribution": float(score),
            "abs_attribution": float(abs(score)),
            "target_type": target,
            "target_label": labels[target_id],
            "pred": labels[pred_id],
            "gold": labels[gold_id],
            "pred_confidence": float(probs[pred_id]),
            "gold_probability": float(probs[gold_id]),
        })

    if was_training:
        model.train()
    return rows
