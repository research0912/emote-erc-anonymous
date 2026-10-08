from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


CONFIGS = [
    ("No text", "configs/meld_roberta_large_v10_ablate_no_text_a100.yaml"),
    ("No dialogue-flow", "configs/meld_roberta_large_v10_ablate_no_dialogue_flow_a100.yaml"),
    ("No speaker-state", "configs/meld_roberta_large_v10_ablate_no_speaker_state_a100.yaml"),
    ("No emotion-shift", "configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100.yaml"),
    ("No cross-attention", "configs/meld_roberta_large_v10_ablate_no_cross_attention_a100.yaml"),
    ("No GNN", "configs/meld_roberta_large_v10_ablate_no_gnn_a100.yaml"),
]


def _output_dir(config_path: str) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg["experiment"]["output_dir"])


def main() -> None:
    rows = []
    for variant, config_path in CONFIGS:
        out_dir = _output_dir(config_path)
        metrics_path = out_dir / "test_metrics.json"
        if not metrics_path.exists():
            print(f"\n=== Training {variant}: {config_path} ===", flush=True)
            subprocess.run([sys.executable, "train.py", "--config", config_path], check=True)
        else:
            print(f"\n=== Skipping {variant}; found {metrics_path} ===", flush=True)

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({
            "variant": variant,
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        })

    full_metrics_path = Path("outputs/meld_roberta_large_v10_refined_a100/test_metrics.json")
    full_weighted = None
    if full_metrics_path.exists():
        with open(full_metrics_path, "r", encoding="utf-8") as f:
            full_metrics = json.load(f)
        full_weighted = full_metrics["weighted_f1"]
        rows.insert(0, {
            "variant": "Full model",
            "macro_f1": full_metrics["macro_f1"],
            "weighted_f1": full_metrics["weighted_f1"],
        })

    for row in rows:
        row["delta_weighted_f1"] = "" if full_weighted is None or row["variant"] == "Full model" else row["weighted_f1"] - full_weighted

    out_path = Path("outputs/v10_training_ablation_a100_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "macro_f1", "weighted_f1", "delta_weighted_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary to {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
