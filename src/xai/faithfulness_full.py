from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
from .context_masking import context_masking_analysis
from .speaker_state_masking import speaker_state_masking_analysis
from .emotion_shift_analysis import emotion_shift_masking_analysis
from .token_occlusion import simple_token_occlusion
from .integrated_gradients import integrated_gradients_main_input


def full_faithfulness_suite(model, tokenizer, samples: List[Dict[str, Any]], labels: List[str], device: str, max_length: int, out_dir, max_examples: int = 50, max_tokens: int = 25, run_integrated_gradients: bool = True, ig_steps: int = 24):
    """Runs all faithfulness checks and writes paper-ready CSVs.

    Outputs:
    - context_masking.csv
    - speaker_state_masking.csv
    - emotion_shift_masking.csv
    - token_occlusion.csv
    - xai_summary.csv
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context_rows, speaker_rows, shift_rows, token_rows, ig_rows, summary_rows = [], [], [], [], [], []

    for sample in samples[:max_examples]:
        base, ctx_df = context_masking_analysis(model, tokenizer, sample, labels, device, max_length)
        spk_df = speaker_state_masking_analysis(model, tokenizer, sample, labels, device, max_length)
        shf_df = emotion_shift_masking_analysis(model, tokenizer, sample, labels, device, max_length)
        tok_rows = simple_token_occlusion(model, tokenizer, sample, labels, device, max_length, max_tokens=max_tokens)
        if run_integrated_gradients:
            sample_ig_rows = integrated_gradients_main_input(
                model, tokenizer, sample, labels, device, max_length, steps=ig_steps, target="pred"
            )
        else:
            sample_ig_rows = []

        if not ctx_df.empty:
            context_rows.extend(ctx_df.to_dict("records"))
        speaker_rows.extend(spk_df.to_dict("records"))
        shift_rows.extend(shf_df.to_dict("records"))
        token_rows.extend(tok_rows)
        for row in sample_ig_rows:
            ig_rows.append({
                "dialogue_id": sample.get("dialogue_id"),
                "turn_id": sample.get("turn_id"),
                "speaker": sample.get("speaker"),
                "utterance": sample.get("utterance"),
                **row,
            })

        top_context_drop = float(ctx_df["gold_probability_drop"].max()) if not ctx_df.empty else 0.0
        speaker_drop = float(spk_df.iloc[0]["gold_probability_drop"]) if not spk_df.empty else 0.0
        shift_drop = float(shf_df.iloc[0]["gold_probability_drop"]) if not shf_df.empty else 0.0
        summary_rows.append({
            "dialogue_id": sample.get("dialogue_id"),
            "turn_id": sample.get("turn_id"),
            "speaker": sample.get("speaker"),
            "gold": sample.get("label_name"),
            "utterance": sample.get("utterance"),
            "prediction": base["pred"],
            "prediction_confidence": base["confidence"],
            "top_context_drop": top_context_drop,
            "speaker_state_drop": speaker_drop,
            "emotion_shift_drop": shift_drop,
            "dominant_explanation_source": max(
                [("context", top_context_drop), ("speaker_state", speaker_drop), ("emotion_shift", shift_drop)],
                key=lambda x: x[1],
            )[0],
        })

    pd.DataFrame(context_rows).to_csv(out_dir / "context_masking.csv", index=False)
    pd.DataFrame(speaker_rows).to_csv(out_dir / "speaker_state_masking.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out_dir / "emotion_shift_masking.csv", index=False)
    pd.DataFrame(token_rows).to_csv(out_dir / "token_occlusion.csv", index=False)
    pd.DataFrame(ig_rows).to_csv(out_dir / "integrated_gradients.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "xai_summary.csv", index=False)
    return pd.DataFrame(summary_rows)
