import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["experiment"]["output_dir"])
    latest_file = out_dir / "best_model" / "latest_checkpoint.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"Missing best checkpoint pointer: {latest_file}")

    checkpoint = latest_file.read_text(encoding="utf-8").strip()
    subprocess.run(
        [
            sys.executable,
            "evaluate.py",
            "--config",
            args.config,
            "--checkpoint",
            checkpoint,
            "--split",
            args.split,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
