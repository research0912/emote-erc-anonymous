import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


CONFIGS = [
    ("Full model", "configs/meld_roberta_large_v10_refined_seed43.yaml"),
    ("Current utterance only", "configs/meld_roberta_large_text_only_a100_seed43.yaml"),
    ("No dialogue-structure", "configs/meld_roberta_large_v10_ablate_no_dialogue_structure_a100_seed43.yaml"),
    ("No speaker-state", "configs/meld_roberta_large_v10_ablate_no_speaker_state_a100_seed43.yaml"),
    ("No emotion-shift", "configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100_seed43.yaml"),
    ("Full model", "configs/meld_roberta_large_v10_refined_seed44.yaml"),
    ("Current utterance only", "configs/meld_roberta_large_text_only_a100_seed44.yaml"),
    ("No dialogue-structure", "configs/meld_roberta_large_v10_ablate_no_dialogue_structure_a100_seed44.yaml"),
    ("No speaker-state", "configs/meld_roberta_large_v10_ablate_no_speaker_state_a100_seed44.yaml"),
    ("No emotion-shift", "configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100_seed44.yaml"),
]


def read_experiment_fields(config_path: Path) -> dict:
    fields = {}
    in_experiment = False
    with config_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if line == "experiment:":
                in_experiment = True
                continue
            if line and not line.startswith(" "):
                in_experiment = False
            if in_experiment and line.startswith("  name: "):
                fields["name"] = line.split(": ", 1)[1]
            elif in_experiment and line.startswith("  seed: "):
                fields["seed"] = int(line.split(": ", 1)[1])
            elif in_experiment and line.startswith("  output_dir: "):
                fields["output_dir"] = line.split(": ", 1)[1]
    return fields


def replace_yaml_scalar(text: str, key: str, value: int) -> str:
    lines = text.splitlines(keepends=True)
    prefix = f"  {key}: "
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            newline = "\n" if line.endswith("\n") else ""
            lines[i] = f"{prefix}{value}{newline}"
            break
    return "".join(lines)


def patch_batch(config_path: Path, batch_size: Optional[int], grad_accum_steps: Optional[int]) -> None:
    if batch_size is None and grad_accum_steps is None:
        return
    text = config_path.read_text(encoding="utf-8")
    if batch_size is not None:
        text = replace_yaml_scalar(text, "batch_size", batch_size)
    if grad_accum_steps is not None:
        text = replace_yaml_scalar(text, "grad_accum_steps", grad_accum_steps)
    config_path.write_text(text, encoding="utf-8")


def test_metrics_path(project_root: Path, config_name: str) -> Path:
    fields = read_experiment_fields(project_root / config_name)
    out_dir = Path(fields["output_dir"])
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    return out_dir / "test_metrics.json"


def write_summaries(project_root: Path) -> None:
    rows = []
    grouped = {}
    for variant, config_name in CONFIGS:
        fields = read_experiment_fields(project_root / config_name)
        metrics_path = test_metrics_path(project_root, config_name)
        row = {
            "variant": variant,
            "experiment": fields["name"],
            "seed": fields["seed"],
            "config": config_name,
            "status": "missing",
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "loss": "",
        }
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row.update({
                "status": "done",
                "accuracy": metrics.get("accuracy", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
                "loss": metrics.get("loss", ""),
            })
            grouped.setdefault(variant, []).append(row)
        rows.append(row)

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "paper_ablation_seed43_44_summary.csv"
    mean_path = out_dir / "paper_ablation_seed43_44_mean_std.csv"

    fields = ["variant", "experiment", "seed", "config", "status", "accuracy", "macro_f1", "weighted_f1", "loss"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    mean_fields = ["variant", "num_seeds", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "weighted_f1_std"]
    mean_rows = []
    for variant, variant_rows in grouped.items():
        row = {"variant": variant, "num_seeds": len(variant_rows)}
        for metric in ["accuracy", "macro_f1", "weighted_f1"]:
            vals = [float(r[metric]) for r in variant_rows if r[metric] != ""]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = var ** 0.5 if len(vals) > 1 else 0.0
        mean_rows.append(row)
    with mean_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fields)
        writer.writeheader()
        writer.writerows(mean_rows)

    print(f"Saved per-run summary: {summary_path}")
    print(f"Saved mean/std summary: {mean_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, choices=[43, 44], default=None, help="Run only one seed.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    configs = CONFIGS
    if args.seed is not None:
        configs = [(label, cfg) for label, cfg in CONFIGS if f"_seed{args.seed}.yaml" in cfg]

    if not args.summary_only:
        for _, config_name in configs:
            patch_batch(project_root / config_name, args.batch_size, args.grad_accum_steps)
            if args.skip_existing and test_metrics_path(project_root, config_name).exists():
                print(f"Skipping completed config: {config_name}", flush=True)
                continue
            cmd = [sys.executable, "train.py", "--config", config_name]
            if args.resume and (test_metrics_path(project_root, config_name).parent / "latest_checkpoint.txt").exists():
                cmd.extend(["--resume", "latest"])
            print("=" * 100, flush=True)
            print(f"Running: {' '.join(cmd)}", flush=True)
            print("=" * 100, flush=True)
            subprocess.run(cmd, cwd=project_root, check=True)

    write_summaries(project_root)


if __name__ == "__main__":
    main()
