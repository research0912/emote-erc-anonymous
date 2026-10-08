import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def run(cmd):
    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_encoder_metrics(config_path):
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["output_dir"])
    metrics_path = out_dir / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file after {cfg_path}: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "config": str(cfg_path),
        "experiment": cfg["experiment"].get("name", cfg_path.stem),
        "encoder": cfg["model"].get("base_model", cfg_path.stem),
        "output_dir": str(out_dir),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def save_encoder_summary(rows, out_dir="outputs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out / "encoder_comparison.csv"
    png_path = out / "encoder_comparison.png"
    df.to_csv(csv_path, index=False)

    plot_df = df.set_index("encoder")[["accuracy", "macro_f1", "weighted_f1"]]
    ax = plot_df.plot(kind="bar", figsize=(8, 4.5))
    ax.set_xlabel("Encoder")
    ax.set_ylabel("Score")
    ax.set_title("Encoder Comparison")
    ax.legend(title="Metric")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved {csv_path}")
    print(f"Saved {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multiple encoder configs sequentially.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()
    rows = []
    for cfg in args.configs:
        if not args.skip_train:
            run(["python", "train.py", "--config", cfg])
        rows.append(read_encoder_metrics(cfg))
        print(f"Finished config: {cfg}")
    save_encoder_summary(rows)


if __name__ == "__main__":
    main()
