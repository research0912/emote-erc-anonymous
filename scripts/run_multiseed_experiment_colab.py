"""
Multi-seed experiment runner for Google Colab.
Handles Drive mounting, path resolution, and batch seed execution.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def setup_colab():
    """Mount Google Drive if running in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("Google Drive mounted successfully.")
        return True
    except ImportError:
        print("Not running in Colab, skipping Drive mount.")
        return False


def run_single_seed(config_path: str, seed: int, output_tag: str, project_root: Path = None) -> Dict:
    """Run training with a single seed and return the results path."""
    config_path = Path(config_path).resolve()
    
    # Load and modify config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    cfg["experiment"]["seed"] = seed
    cfg["experiment"]["output_dir"] = f"outputs/meld_roberta_{output_tag}_seed_{seed}"
    
    # Ensure output dir path is absolute for Colab
    if project_root:
        output_abs = project_root / cfg["experiment"]["output_dir"]
        cfg["experiment"]["output_dir"] = str(output_abs)
    
    # Write temporary config
    if project_root:
        temp_config_dir = project_root / "configs"
    else:
        temp_config_dir = Path("configs")
    
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config = temp_config_dir / f"_temp_seed_{seed}.yaml"
    
    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    
    print(f"\n{'='*80}")
    print(f"Running seed {seed} (output: {cfg['experiment']['output_dir']})")
    print(f"{'='*80}")
    
    # Run training
    if project_root:
        cmd = [sys.executable, str(project_root / "train.py"), "--config", str(temp_config)]
    else:
        cmd = [sys.executable, "train.py", "--config", str(temp_config)]
    
    result = subprocess.run(cmd, cwd=project_root or Path.cwd())
    
    # Clean up temp config
    temp_config.unlink()
    
    if result.returncode != 0:
        print(f"ERROR: Seed {seed} failed with return code {result.returncode}")
        return None
    
    # Return path to results
    return {
        "seed": seed,
        "output_dir": cfg["experiment"]["output_dir"],
    }


def load_metrics(output_dir: str) -> Dict:
    """Load val/test metrics from a run."""
    output_path = Path(output_dir)
    metrics = {}
    
    for split in ["val", "test"]:
        metrics_file = output_path / f"{split}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics[split] = json.load(f)
        else:
            print(f"WARNING: {metrics_file} not found")
            metrics[split] = None
    
    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across seeds."""
    aggregated = {}
    
    for split in ["val", "test"]:
        split_metrics_list = [m[split] for m in all_metrics if m.get(split) is not None]
        if not split_metrics_list:
            continue
        
        aggregated[split] = {}
        
        # Get all metric keys from first result
        all_keys = set()
        for m in split_metrics_list:
            all_keys.update(m.keys())
        
        # Filter out non-numeric metrics
        numeric_keys = [k for k in all_keys if k not in ["classification_report_text", "predictions"]]
        
        for key in numeric_keys:
            values = []
            for m in split_metrics_list:
                if key in m and isinstance(m[key], (int, float)):
                    values.append(m[key])
            
            if values:
                aggregated[split][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed experiment on Colab")
    parser.add_argument("--config", required=True, help="Base config path (can be absolute or relative)")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated seed values")
    parser.add_argument("--output-tag", default="v8_multiseed_colab", help="Tag for output directory names")
    parser.add_argument("--project-root", default=None, help="Absolute project root (for Colab)")
    args = parser.parse_args()
    
    # Setup Colab if needed
    is_colab = setup_colab()
    
    # Resolve project root for Colab
    project_root = None
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Running experiment with seeds: {seeds}")
    print(f"Project root: {project_root or 'current directory'}")
    
    results = []
    all_metrics = []
    
    # Run each seed
    for seed in seeds:
        result = run_single_seed(args.config, seed, args.output_tag, project_root)
        if result:
            results.append(result)
            metrics = load_metrics(result["output_dir"])
            all_metrics.append(metrics)
        else:
            print(f"Skipping seed {seed} due to error")
    
    if not all_metrics:
        print("ERROR: No successful runs")
        return
    
    # Aggregate results
    aggregated = aggregate_metrics(all_metrics)
    
    # Create summary directory
    if project_root:
        summary_dir = project_root / f"outputs/multiseed_summary_{args.output_tag}"
    else:
        summary_dir = Path(f"outputs/multiseed_summary_{args.output_tag}")
    
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated metrics
    summary_file = summary_dir / "aggregated_metrics.json"
    with open(summary_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n\n{'='*80}")
    print("MULTISEED SUMMARY")
    print(f"{'='*80}\n")
    
    # Print summary
    for split in ["val", "test"]:
        if split not in aggregated:
            continue
        print(f"\n{split.upper()} METRICS (mean ± std):")
        print("-" * 60)
        
        for key, stats in sorted(aggregated[split].items()):
            if "values" in stats:
                print(f"  {key:30s}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
                      f"[min={stats['min']:.4f}, max={stats['max']:.4f}]")
    
    # Save detailed breakdown
    breakdown_file = summary_dir / "detailed_breakdown.txt"
    with open(breakdown_file, "w") as f:
        f.write(f"Multi-Seed Experiment Summary\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Output tag: {args.output_tag}\n")
        f.write(f"Project root: {project_root or 'current directory'}\n\n")
        
        for split in ["val", "test"]:
            if split not in aggregated:
                continue
            f.write(f"\n{split.upper()} BREAKDOWN:\n")
            f.write("=" * 80 + "\n")
            for key, stats in sorted(aggregated[split].items()):
                f.write(f"\n{key}:\n")
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std:  {stats['std']:.4f}\n")
                f.write(f"  Min:  {stats['min']:.4f}\n")
                f.write(f"  Max:  {stats['max']:.4f}\n")
                f.write(f"  Values per seed: {[f'{v:.4f}' for v in stats['values']]}\n")
    
    print(f"\n\nResults saved to: {summary_dir}")
    print(f"  - aggregated_metrics.json")
    print(f"  - detailed_breakdown.txt")


if __name__ == "__main__":
    main()
