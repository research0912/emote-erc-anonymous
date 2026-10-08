import argparse
from pathlib import Path
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer
from train import evaluate, make_loaders
from src.models.context_speaker_shift_model import ContextSpeakerShiftERC
from src.training.utils import get_device, load_checkpoint, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    labels = cfg["dataset"]["labels"]
    id2label = {i: label for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    _, _, _, _, val_loader, test_loader = make_loaders(cfg, tokenizer)
    loader = val_loader if args.split == "val" else test_loader
    device = get_device()
    model = ContextSpeakerShiftERC(cfg, len(labels)).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    out_dir = Path(cfg["experiment"]["output_dir"]) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    ablations = [
        ("full", None),
        ("no_context", "no_context"),
        ("no_speaker_state", "no_speaker_state"),
        ("no_shift", "no_shift"),
        ("no_dialogue_flow", "no_dialogue_flow"),
        ("no_bilstm", "no_bilstm"),
    ]
    for name, ablation in ablations:
        metrics = evaluate(model, loader, device, id2label, ablation=ablation)
        metrics.pop("predictions")
        save_json(metrics, out_dir / f"{args.split}_{name}_metrics.json")
        rows.append({
            "setting": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })
    pd.DataFrame(rows).to_csv(out_dir / f"{args.split}_ablation_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
