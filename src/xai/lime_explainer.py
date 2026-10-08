from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

from src.data_loader.base import build_model_text


def run_lime_text_explanations(model, tokenizer, samples: List[Dict[str, Any]], labels: List[str], device: str, max_length: int, out_csv: str, max_examples: int = 20, num_features: int = 10):
    """LIME local token explanations for selected ERC examples."""
    if LimeTextExplainer is None:
        pd.DataFrame([{"status": "LIME not installed. Install lime or skip this module."}]).to_csv(out_csv, index=False)
        return

    import torch
    model.eval()

    def predict_proba(texts):
        enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], ablation="no_speaker_state")
            return out["logits"].softmax(-1).detach().cpu().numpy()

    explainer = LimeTextExplainer(class_names=labels)
    rows = []
    for sample in samples[:max_examples]:
        text = build_model_text(sample, include_context=True)
        try:
            exp = explainer.explain_instance(text, predict_proba, num_features=num_features, labels=list(range(len(labels))))
            pred_label = exp.available_labels()[0] if exp.available_labels() else None
            features = exp.as_list(label=pred_label) if pred_label is not None else []
            for rank, (token, weight) in enumerate(features):
                rows.append({
                    "dialogue_id": sample.get("dialogue_id"),
                    "turn_id": sample.get("turn_id"),
                    "gold": sample.get("label_name"),
                    "rank": rank + 1,
                    "token_or_phrase": token,
                    "weight": weight,
                    "explained_label": labels[pred_label] if pred_label is not None else None,
                    "text": text,
                })
        except Exception as exc:
            rows.append({"dialogue_id": sample.get("dialogue_id"), "turn_id": sample.get("turn_id"), "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
