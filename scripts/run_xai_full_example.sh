#!/bin/bash

CKPT=$(cat outputs/masac_xlmr/best_model/latest_checkpoint.txt)

python run_xai.py \
  --config configs/masac_xlmr.yaml \
  --checkpoint "$CKPT" \
  --split test \
  --num_examples 50 \
  --max_tokens 25
