from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIGS = [
    "configs/meld_roberta_base_text_only_a100.yaml",
    "configs/meld_roberta_large_text_only_a100.yaml",
    "configs/meld_roberta_large_v10_ablate_no_text_a100.yaml",
]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_config(config_path: Path, force: bool = False) -> dict:
    cfg = load_yaml(config_path)
    metrics_path = Path(cfg["experiment"]["output_dir"]) / "test_metrics.json"
    if force or not metrics_path.exists():
        print(f"\n=== Training {config_path} ===", flush=True)
        subprocess.run([sys.executable, "train.py", "--config", str(config_path)], check=True)
    else:
        print(f"\n=== Skipping {config_path}; found {metrics_path} ===", flush=True)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {
        "experiment": cfg["experiment"]["name"],
        "config": str(config_path),
        "seed": cfg["experiment"]["seed"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def make_seed_config(base_config: Path, seed: int, temp_dir: Path) -> Path:
    cfg = deepcopy(load_yaml(base_config))
    base_name = cfg["experiment"]["name"]
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["name"] = f"{base_name}_seed{seed}"
    cfg["experiment"]["output_dir"] = f"outputs/{base_name}_seed{seed}"
    out_path = temp_dir / f"{base_config.stem}_seed{seed}.yaml"
    write_yaml(out_path, cfg)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="Config files to run once.",
    )
    parser.add_argument(
        "--multiseed-config",
        default="configs/meld_roberta_large_v10_refined.yaml",
        help="Base config used for multiple-seed variance runs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 43, 44],
        help="Seeds for the multi-seed experiment.",
    )
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-multiseed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = []

    if not args.skip_single:
        for config in args.configs:
            rows.append(run_config(Path(config), force=args.force))

    if not args.skip_multiseed:
        temp_dir = Path("configs")
        for seed in args.seeds:
            seed_config = make_seed_config(Path(args.multiseed_config), seed, temp_dir)
            rows.append(run_config(seed_config, force=args.force))

    out_path = Path("outputs/reviewer_experiments_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "config", "seed", "accuracy", "macro_f1", "weighted_f1"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary to {out_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
