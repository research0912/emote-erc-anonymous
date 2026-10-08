from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

try:
    import shap
except Exception:
    shap = None

from src.data_loader.base import build_model_text


def run_shap_text_explanations(model, tokenizer, samples: List[Dict[str, Any]], labels: List[str], device: str, max_length: int, out_csv: str, max_examples: int = 20):
    """SHAP text explanations.

    This is intentionally lightweight. For large datasets, run on selected qualitative examples only.
    If SHAP is unavailable or fails, writes an explanatory fallback CSV instead of crashing.
    """
    rows = []
    if shap is None:
        pd.DataFrame([{"status": "SHAP not installed. Install shap or skip this module."}]).to_csv(out_csv, index=False)
        return

    model.eval()

    def predict_proba(texts):
        import torch
        enc = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            # speaker-state/shift are intentionally disabled for token-only SHAP.
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], ablation="no_speaker_state")
            probs = out["logits"].softmax(-1).detach().cpu().numpy()
        return probs

    texts = [build_model_text(s, include_context=True) for s in samples[:max_examples]]
    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_proba, masker, output_names=labels)
        shap_values = explainer(texts)
        for i, text in enumerate(texts):
            pred_idx = int(shap_values.values[i].mean(axis=0).argmax()) if getattr(shap_values, "values", None) is not None else -1
            rows.append({
                "dialogue_id": samples[i].get("dialogue_id"),
                "turn_id": samples[i].get("turn_id"),
                "gold": samples[i].get("label_name"),
                "text": text,
                "note": "SHAP values computed; use notebook visualization for token plots.",
                "approx_top_class_index": pred_idx,
            })
    except Exception as exc:
        rows.append({"status": "SHAP failed", "error": str(exc)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
