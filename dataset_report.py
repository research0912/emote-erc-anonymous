import argparse
from pathlib import Path
import yaml
from src.data_loader.factory import build_dataset
from src.analysis.dataset_stats import dataset_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    label2id = {label: i for i, label in enumerate(cfg["dataset"]["labels"])}
    out_root = Path(cfg["experiment"]["output_dir"]) / "dataset_report"
    for split in ["train", "val", "test"]:
        ds = build_dataset(
            cfg["dataset"]["name"], cfg["dataset"][f"{split}_path"], label2id,
            cfg["dataset"]["context_window"], cfg["dataset"].get("speaker_history_window", cfg["dataset"]["context_window"])
        )
        dataset_distribution(ds, out_root / split)
        print(f"{split}: {len(ds)} samples")


if __name__ == "__main__":
    main()
