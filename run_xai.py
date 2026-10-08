import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from src.data_loader.factory import build_dataset
from src.models.context_speaker_shift_model import ContextSpeakerShiftERC
from src.training.utils import get_device, load_checkpoint
from src.xai.faithfulness import context_turn_importance, speaker_state_importance, shift_importance
from src.xai.integrated_gradients import integrated_gradients_main_input
from src.xai.token_occlusion import simple_token_occlusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=25)
    parser.add_argument("--ig_steps", type=int, default=24)
    parser.add_argument("--skip_ig", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    labels = cfg["dataset"]["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    path = cfg["dataset"][f"{args.split}_path"]
    ds = build_dataset(
        cfg["dataset"]["name"], path, label2id,
        cfg["dataset"]["context_window"], cfg["dataset"].get("speaker_history_window", cfg["dataset"]["context_window"])
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    device = get_device()
    model = ContextSpeakerShiftERC(cfg, len(labels)).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    out_dir = Path(cfg["experiment"]["output_dir"]) / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    context_rows, speaker_rows, shift_rows, token_rows, ig_rows, summary_rows = [], [], [], [], [], []

    for sample in ds.samples[:args.num_examples]:
        base, ctx = context_turn_importance(model, tokenizer, sample, labels, device, cfg["dataset"]["max_length"])
        spk = speaker_state_importance(model, tokenizer, sample, labels, device, cfg["dataset"]["max_length"])
        shf = shift_importance(model, tokenizer, sample, labels, device, cfg["dataset"]["max_length"])
        tok = simple_token_occlusion(model, tokenizer, sample, labels, device, cfg["dataset"]["max_length"], args.max_tokens)
        ig = [] if args.skip_ig else integrated_gradients_main_input(
            model, tokenizer, sample, labels, device, cfg["dataset"]["max_length"], steps=args.ig_steps, target="pred"
        )

        key = {"dialogue_id": sample["dialogue_id"], "turn_id": sample["turn_id"], "gold": sample["label_name"], "utterance": sample["utterance"]}
        for row in ctx:
            context_rows.append({**key, **row})
        speaker_rows.append({**key, **spk})
        shift_rows.append({**key, **shf})
        for row in tok:
            token_rows.append({**key, **row})
        for row in ig:
            ig_rows.append({**key, **row})
        summary_rows.append({
            **key,
            "prediction": base["pred"],
            "prediction_confidence": base["confidence"],
            "top_context_drop": max([r["gold_probability_drop"] for r in ctx], default=0.0),
            "speaker_state_drop": spk["gold_probability_drop"],
            "shift_drop": shf["gold_probability_drop"],
        })

    pd.DataFrame(context_rows).to_csv(out_dir / "context_turn_importance.csv", index=False)
    pd.DataFrame(speaker_rows).to_csv(out_dir / "speaker_state_importance.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(out_dir / "emotion_shift_importance.csv", index=False)
    pd.DataFrame(token_rows).to_csv(out_dir / "token_occlusion_importance.csv", index=False)
    pd.DataFrame(ig_rows).to_csv(out_dir / "integrated_gradients.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "xai_summary.csv", index=False)
    print(f"Saved XAI outputs to {out_dir}")


if __name__ == "__main__":
    main()
