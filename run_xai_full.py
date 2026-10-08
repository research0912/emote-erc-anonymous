from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer

from src.data_loader.factory import build_dataset
from src.models.context_speaker_shift_model import ContextSpeakerShiftERC
from src.training.utils import get_device, load_checkpoint
from src.xai.faithfulness_full import full_faithfulness_suite
from src.xai.shap_explainer import run_shap_text_explanations
from src.xai.lime_explainer import run_lime_text_explanations
from src.xai.emotion_shift_analysis import transition_matrix_from_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=25)
    parser.add_argument("--ig_steps", type=int, default=24)
    parser.add_argument("--skip_ig", action="store_true")
    parser.add_argument("--run_shap", action="store_true")
    parser.add_argument("--run_lime", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    labels = cfg["dataset"]["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    path = cfg["dataset"][f"{args.split}_path"]
    ds = build_dataset(
        cfg["dataset"]["name"], path, label2id,
        cfg["dataset"].get("context_window", 5),
        cfg["dataset"].get("speaker_history_window", cfg["dataset"].get("context_window", 5)),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    device = get_device()
    model = ContextSpeakerShiftERC(cfg, len(labels)).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    out_dir = Path(cfg["experiment"]["output_dir"]) / "xai" / f"{args.split}_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_faithfulness_suite(
        model=model,
        tokenizer=tokenizer,
        samples=ds.samples,
        labels=labels,
        device=device,
        max_length=cfg["dataset"].get("max_length", 256),
        out_dir=out_dir,
        max_examples=args.num_examples,
        max_tokens=args.max_tokens,
        run_integrated_gradients=not args.skip_ig,
        ig_steps=args.ig_steps,
    )

    transition_matrix_from_samples(ds.samples).to_csv(out_dir / "emotion_transition_matrix.csv")

    if args.run_shap:
        run_shap_text_explanations(
            model, tokenizer, ds.samples, labels, device,
            cfg["dataset"].get("max_length", 256),
            out_csv=str(out_dir / "shap_text_explanations.csv"),
            max_examples=min(args.num_examples, 20),
        )

    if args.run_lime:
        run_lime_text_explanations(
            model, tokenizer, ds.samples, labels, device,
            cfg["dataset"].get("max_length", 256),
            out_csv=str(out_dir / "lime_text_explanations.csv"),
            max_examples=min(args.num_examples, 20),
        )

    print(f"Saved full XAI outputs to {out_dir}")


if __name__ == "__main__":
    main()
