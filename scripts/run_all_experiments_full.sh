#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
  configs/meld_roberta.yaml
  configs/meld_deberta.yaml
  configs/meld_xlmr.yaml
  configs/masac_roberta.yaml
  configs/masac_deberta.yaml
  configs/masac_xlmr.yaml
)

for CONFIG in "${CONFIGS[@]}"; do
  echo
  echo "Running full pipeline for $CONFIG"
  bash scripts/run_config_full.sh "$CONFIG"
done

echo
echo "All experiment ZIP files:"
ls -lh outputs/*.zip 2>/dev/null || true
