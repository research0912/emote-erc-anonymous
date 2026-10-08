#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: scripts/run_config_full.sh <config.yaml>}
NUM_EXAMPLES=${NUM_EXAMPLES:-50}
MAX_TOKENS=${MAX_TOKENS:-25}

OUT_DIR=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))["experiment"]["output_dir"])' "$CONFIG")

python dataset_report.py --config "$CONFIG"
python train.py --config "$CONFIG"

CKPT=$(cat "$OUT_DIR/best_model/latest_checkpoint.txt")

python run_ablation.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --split test

python run_xai.py \
  --config "$CONFIG" \
  --checkpoint "$CKPT" \
  --split test \
  --num_examples "$NUM_EXAMPLES" \
  --max_tokens "$MAX_TOKENS"

python plot_results.py \
  --metrics "$OUT_DIR/test_metrics.json" \
  --ablation_csv "$OUT_DIR/ablation/test_ablation_summary.csv" \
  --label_distribution_csv "$OUT_DIR/dataset_report/test/label_distribution.csv" \
  --transition_csv "$OUT_DIR/dataset_report/test/emotion_transition_matrix.csv" \
  --xai_summary_csv "$OUT_DIR/xai/xai_summary.csv" \
  --context_importance_csv "$OUT_DIR/xai/context_turn_importance.csv" \
  --speaker_state_csv "$OUT_DIR/xai/speaker_state_importance.csv" \
  --out_dir "$OUT_DIR/plots"

# Keep the output folder small before backup:
# - keep only the latest regular training checkpoint
# - keep only the latest best_model checkpoint
# - remove intermediate validation metric snapshots
find "$OUT_DIR" -maxdepth 1 -type d -name "checkpoint-step-*" \
  | sort -V \
  | head -n -1 \
  | xargs -r rm -rf

if [ -d "$OUT_DIR/best_model" ]; then
  find "$OUT_DIR/best_model" -maxdepth 1 -type d -name "checkpoint-step-*" \
    | sort -V \
    | head -n -1 \
    | xargs -r rm -rf
fi

find "$OUT_DIR" -name "val_metrics_step_*.json" -type f -delete

ZIP_PATH="${OUT_DIR}.zip"
rm -f "$ZIP_PATH"
zip -r "$ZIP_PATH" "$OUT_DIR"

echo
echo "Saved compressed experiment output to: $ZIP_PATH"
echo "ZIP size:"
ls -lh "$ZIP_PATH"

DOWNLOAD_HELPER="${OUT_DIR}_download.py"
cat > "$DOWNLOAD_HELPER" <<PY
from pathlib import Path

zip_path = Path("$ZIP_PATH")
if not zip_path.exists():
    raise FileNotFoundError(zip_path)

try:
    from google.colab import files
except ImportError:
    print(f"Download manually from: {zip_path}")
else:
    files.download(str(zip_path))
PY

echo "Notebook download helper written to: $DOWNLOAD_HELPER"
echo "In Colab/Jupyter, run:"
echo "python $DOWNLOAD_HELPER"
