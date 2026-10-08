import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


CONFIGS = [
    ("Full model", "configs/meld_roberta_large_v10_refined_seed43.yaml"),
    ("No dialogue-flow", "configs/meld_roberta_large_v10_ablate_no_dialogue_flow_a100_seed43.yaml"),
    ("No speaker-state", "configs/meld_roberta_large_v10_ablate_no_speaker_state_a100_seed43.yaml"),
    ("No emotion-shift", "configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100_seed43.yaml"),
    ("No cross-attention", "configs/meld_roberta_large_v10_ablate_no_cross_attention_a100_seed43.yaml"),
    ("No GNN", "configs/meld_roberta_large_v10_ablate_no_gnn_a100_seed43.yaml"),
    ("No text", "configs/meld_roberta_large_v10_ablate_no_text_a100_seed43.yaml"),
    ("Full model", "configs/meld_roberta_large_v10_refined_seed44.yaml"),
    ("No dialogue-flow", "configs/meld_roberta_large_v10_ablate_no_dialogue_flow_a100_seed44.yaml"),
    ("No speaker-state", "configs/meld_roberta_large_v10_ablate_no_speaker_state_a100_seed44.yaml"),
    ("No emotion-shift", "configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100_seed44.yaml"),
    ("No cross-attention", "configs/meld_roberta_large_v10_ablate_no_cross_attention_a100_seed44.yaml"),
    ("No GNN", "configs/meld_roberta_large_v10_ablate_no_gnn_a100_seed44.yaml"),
    ("No text", "configs/meld_roberta_large_v10_ablate_no_text_a100_seed44.yaml"),
]


def read_simple_config_fields(config_path: Path) -> dict:
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


def maybe_patch_for_small_gpu(config_path: Path, batch_size: Optional[int], grad_accum_steps: Optional[int]) -> None:
    if batch_size is None and grad_accum_steps is None:
        return

    text = config_path.read_text(encoding="utf-8")
    if batch_size is not None:
        text = replace_yaml_scalar(text, "batch_size", batch_size)
    if grad_accum_steps is not None:
        text = replace_yaml_scalar(text, "grad_accum_steps", grad_accum_steps)
    config_path.write_text(text, encoding="utf-8")


def metric_path(project_root: Path, config_path: Path) -> Path:
    cfg = read_simple_config_fields(project_root / config_path)
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    return out_dir / "test_metrics.json"


def collect_results(project_root: Path, rows_path: Path, mean_path: Path) -> None:
    rows = []
    by_variant: dict[str, list[dict]] = {}

    for variant, config_name in CONFIGS:
        config_path = project_root / config_name
        cfg = read_simple_config_fields(config_path)
        seed = cfg["seed"]
        out_dir = Path(cfg["output_dir"])
        if not out_dir.is_absolute():
            out_dir = project_root / out_dir
        metrics_file = out_dir / "test_metrics.json"

        row = {
            "variant": variant,
            "experiment": cfg["name"],
            "seed": seed,
            "config": config_name,
            "status": "missing",
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "loss": "",
        }

        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            row.update({
                "status": "done",
                "accuracy": metrics.get("accuracy", ""),
                "macro_f1": metrics.get("macro_f1", ""),
                "weighted_f1": metrics.get("weighted_f1", ""),
                "loss": metrics.get("loss", ""),
            })
            by_variant.setdefault(variant, []).append(row)

        rows.append(row)

    rows_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "experiment", "seed", "config", "status", "accuracy", "macro_f1", "weighted_f1", "loss"]
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_rows = []
    for variant, variant_rows in by_variant.items():
        mean_row = {"variant": variant, "num_seeds": len(variant_rows)}
        for metric in ["accuracy", "macro_f1", "weighted_f1", "loss"]:
            values = [float(row[metric]) for row in variant_rows if row[metric] != ""]
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
                mean_row[f"{metric}_mean"] = mean
                mean_row[f"{metric}_std"] = variance ** 0.5 if len(values) > 1 else 0.0
            else:
                mean_row[f"{metric}_mean"] = ""
                mean_row[f"{metric}_std"] = ""
        mean_rows.append(mean_row)

    mean_fields = [
        "variant", "num_seeds",
        "accuracy_mean", "accuracy_std",
        "macro_f1_mean", "macro_f1_std",
        "weighted_f1_mean", "weighted_f1_std",
        "loss_mean", "loss_std",
    ]
    with mean_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fields)
        writer.writeheader()
        writer.writerows(mean_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Repository root. In Colab, pass the Drive project path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for smaller GPUs, e.g. 2 or 1.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Optional override for smaller GPUs, e.g. 8 or 16.")
    parser.add_argument("--resume", action="store_true", help="Resume from each config's latest checkpoint when available.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip configs that already have test_metrics.json.")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild summary CSVs from existing outputs.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    rows_path = project_root / "outputs" / "ablation_seed43_44_summary.csv"
    mean_path = project_root / "outputs" / "ablation_seed43_44_mean_std.csv"

    if not args.summary_only:
        for _, config_name in CONFIGS:
            maybe_patch_for_small_gpu(project_root / config_name, args.batch_size, args.grad_accum_steps)

            if args.skip_existing and metric_path(project_root, Path(config_name)).exists():
                print(f"Skipping completed config: {config_name}", flush=True)
                continue

            cmd = [sys.executable, "train.py", "--config", config_name]
            if args.resume:
                latest_path = metric_path(project_root, Path(config_name)).parent / "latest_checkpoint.txt"
                if latest_path.exists():
                    cmd.extend(["--resume", "latest"])

            print("=" * 100, flush=True)
            print(f"Running: {' '.join(cmd)}", flush=True)
            print("=" * 100, flush=True)
            subprocess.run(cmd, cwd=project_root, check=True)

    collect_results(project_root, rows_path, mean_path)
    print(f"Saved per-run summary: {rows_path}")
    print(f"Saved mean/std summary: {mean_path}")


if __name__ == "__main__":
    main()
