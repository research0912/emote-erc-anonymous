# EMOTE: Faithful Explainability for Emotion Recognition in Conversation through Evidence Decomposition

This repository contains the MELD experiments for **EMOTE**, a text-only
explainable Emotion Recognition in Conversation (ERC) framework. The final
experiments decompose conversational evidence into dialogue-flow, speaker-state,
emotion-shift, graph-context, and cross-attention evidence streams, then evaluate
both prediction performance and explanation behavior.

The current README focuses only on the final MELD experiments used in the paper:

- RoBERTa-base optimized evidence model (`v9`)
- RoBERTa-large proposed model (`v10`)
- V10 training-level ablations
- inference-level ablation
- masking-based faithfulness, token occlusion, and Integrated Gradients

## Repository Layout

```text
configs/
  meld_roberta_v9_optimized_base_a100.yaml
  meld_roberta_large_v10_refined.yaml
  meld_roberta_large_v10_ablate_no_dialogue_flow_a100.yaml
  meld_roberta_large_v10_ablate_no_speaker_state_a100.yaml
  meld_roberta_large_v10_ablate_no_emotion_shift_a100.yaml
  meld_roberta_large_v10_ablate_no_cross_attention_a100.yaml
  meld_roberta_large_v10_ablate_no_gnn_a100.yaml

src/
  data_loader/      MELD loading and turn-level sample construction
  models/           Evidence-stream ERC model
  training/         Collation, metrics, and training utilities
  xai/              Masking, occlusion, faithfulness, Integrated Gradients

train.py            Train and evaluate one config
evaluate.py         Evaluate a saved checkpoint
run_ablation.py     Inference-level ablation for a trained checkpoint
run_xai_full.py     Full XAI and faithfulness runner
scripts/run_v10_training_ablation_a100.py
```

## MELD Data Setup

Create the MELD folder:

```bash
mkdir -p data/meld
```

Place the processed MELD text files here:

```text
data/meld/MELD_train.txt
data/meld/MELD_dev.txt
data/meld/MELD_test.txt
```

The paper experiments use only MELD textual transcripts, speaker identifiers,
and dialogue structure. Audio and visual modalities are not used.

## Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If Colab reports a `torchao` compatibility issue, remove it and restart the
runtime:

```bash
pip uninstall -y torchao
```

For faster Hugging Face downloads, set a token if available:

```bash
export HF_TOKEN=<your_token>
```

In Colab:

```python
import os
os.environ["HF_TOKEN"] = "your_token"
```

## Final MELD Models

### V9: RoBERTa-Base Optimized Evidence Model

Config:

```text
configs/meld_roberta_v9_optimized_base_a100.yaml
```

Key settings:

- Encoder: `roberta-base`
- Context window: 7 previous dialogue turns
- Speaker-history window: 8 previous same-speaker utterances
- Main encoder max length: 256 tokens
- Turn-level context max length: 96 tokens
- LoRA rank 16, alpha 32, dropout 0.1
- LoRA target modules: query, key, value
- Last-four-layer CLS pooling
- Dialogue-flow BiLSTM stream
- GNN graph-context stream
- Speaker-state stream
- Previous-text emotion-shift stream
- Cross-attention, dynamic fusion, attention fusion
- Refined-label cross-entropy and contrastive loss

Run:

```bash
python train.py --config configs/meld_roberta_v9_optimized_base_a100.yaml
```

Main outputs:

```text
outputs/meld_roberta_v9_optimized_base_a100/test_metrics.json
outputs/meld_roberta_v9_optimized_base_a100/test_predictions.csv
outputs/meld_roberta_v9_optimized_base_a100/best_model/
```

### V10: RoBERTa-Large Proposed Model

Config:

```text
configs/meld_roberta_large_v10_refined.yaml
```

Key settings:

- Encoder: `roberta-large`
- Gradient checkpointing enabled
- Context window: 7 previous dialogue turns
- Speaker-history window: 8 previous same-speaker utterances
- Main encoder max length: 256 tokens
- Turn-level context max length: 96 tokens
- LoRA rank 16, alpha 32, dropout 0.1
- LoRA target modules: query, value
- Last-four-layer CLS pooling
- Dialogue-flow BiLSTM stream
- GNN graph-context stream
- Speaker-state stream
- Previous-text emotion-shift stream
- Cross-attention, dynamic fusion, attention fusion
- Refined-label cross-entropy and contrastive loss
- A100 effective batch size: 16

Run:

```bash
python train.py --config configs/meld_roberta_large_v10_refined.yaml
```

Main outputs:

```text
outputs/meld_roberta_large_v10_refined_a100/test_metrics.json
outputs/meld_roberta_large_v10_refined_a100/test_predictions.csv
outputs/meld_roberta_large_v10_refined_a100/best_model/
```

For smaller GPUs such as T4, keep the effective batch size at 16 by lowering the
physical batch size:

```yaml
batch_size: 2
grad_accum_steps: 8
```

If CUDA memory is still insufficient:

```yaml
batch_size: 1
grad_accum_steps: 16
```

## Metrics

Training writes validation and test metrics automatically:

```text
outputs/<experiment>/val_metrics.json
outputs/<experiment>/test_metrics.json
```

Each metrics file includes:

- accuracy
- macro F1
- weighted F1
- per-class precision, recall, and F1
- confusion matrix
- loss

## V10 Training-Level Ablation

Training-level ablation trains a separate model with one evidence component
removed from the beginning. This is used to measure predictive contribution of
each component.

Configs:

```text
configs/meld_roberta_large_v10_ablate_no_dialogue_flow_a100.yaml
configs/meld_roberta_large_v10_ablate_no_speaker_state_a100.yaml
configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100.yaml
configs/meld_roberta_large_v10_ablate_no_cross_attention_a100.yaml
configs/meld_roberta_large_v10_ablate_no_gnn_a100.yaml
```

Run the three main paper ablations:

```bash
python train.py --config configs/meld_roberta_large_v10_ablate_no_dialogue_flow_a100.yaml
python train.py --config configs/meld_roberta_large_v10_ablate_no_speaker_state_a100.yaml
python train.py --config configs/meld_roberta_large_v10_ablate_no_emotion_shift_a100.yaml
```

Run all V10 ablations and generate a summary CSV:

```bash
python scripts/run_v10_training_ablation_a100.py
```

Summary output:

```text
outputs/v10_training_ablation_a100_summary.csv
```

## Inference-Level Ablation

Inference-level ablation evaluates a trained full model while masking/removing
one component at test time. This measures test-time sensitivity of the trained
model, not training-time contribution.

For V10:

```bash
CKPT=$(cat outputs/meld_roberta_large_v10_refined_a100/best_model/latest_checkpoint.txt)

python run_ablation.py \
  --config configs/meld_roberta_large_v10_refined.yaml \
  --checkpoint "$CKPT" \
  --split test
```

For V9:

```bash
CKPT=$(cat outputs/meld_roberta_v9_optimized_base_a100/best_model/latest_checkpoint.txt)

python run_ablation.py \
  --config configs/meld_roberta_v9_optimized_base_a100.yaml \
  --checkpoint "$CKPT" \
  --split test
```

Outputs:

```text
outputs/<experiment>/ablation/test_ablation_summary.csv
outputs/<experiment>/ablation/test_full_metrics.json
outputs/<experiment>/ablation/test_no_context_metrics.json
outputs/<experiment>/ablation/test_no_speaker_state_metrics.json
outputs/<experiment>/ablation/test_no_shift_metrics.json
outputs/<experiment>/ablation/test_no_dialogue_flow_metrics.json
outputs/<experiment>/ablation/test_no_bilstm_metrics.json
```

## Advanced Explainability (XAI) and Faithfulness

The repository includes a comprehensive, multi-modal XAI framework that analyzes model predictions through multiple lenses. The pipeline is executed via `run_xai_advanced.py` and outputs rich JSON analytics suitable for visual dashboards.

### Core XAI Modules
Located in `src/xai/`, the advanced pipeline supports:
- **Stream Attribution**: Quantifies the exact causal contribution (confidence drop) of the `text`, `dialogue_flow`, `speaker_state`, and `emotion_shift` streams using zero-embedding ablation.
- **Explanation Consistency**: Computes bounded consistency metrics (0.0 to 1.0) to ensure the model relies on similar evidence for similar predictions.
- **Emotion Prototypes**: Extracts latent representations and calculates Euclidean distances to emotion class centroids.
- **Evidence Conflict**: Detects scenarios where different evidence streams vote for conflicting emotions.
- **Natural Language Explainer**: Automatically generates human-readable text summaries describing *why* the model made a prediction based on attribution values.
- **Turning Point & Counterfactual Analysis**: Identifies specific historical context turns that causally flipped the model's prediction.
- **Token Occlusion**: Determines the most influential n-grams within the local utterance.

### Running the Advanced XAI Pipeline

To run the full suite of advanced analytics on the V10 model:

```bash
CKPT=$(cat outputs/meld_roberta_large_v10_refined_a100/best_model/latest_checkpoint.txt)

python run_xai_advanced.py \
  --config configs/meld_roberta_large_v10_refined.yaml \
  --checkpoint "$CKPT" \
  --split test \
  --num_examples 100 \
  --max_tokens 20
```

### XAI Outputs

Instead of fragmented CSVs, the advanced pipeline generates unified JSON analytics designed to power Jupyter Notebook dashboards (e.g., `Colab_Advanced_XAI.ipynb`):

```text
outputs/<experiment>/xai/test_advanced/xai_dashboard_advanced.json
```

### Visual Dashboards

The repository includes tools for rendering academic visualizations (Evidence Radar Charts, Prototype MDS, Confidence Distributions, etc.):
- Use the included `erc_paper_results_final.ipynb` or `xai_dashboard_analytics.ipynb` to render publication-ready visuals from the JSON output.

## Reproducibility Notes

- The final experiments use MELD only.
- The final experiments are text-only.
- Gold previous emotion labels are not used.
- The emotion-shift stream uses previous utterance text and current utterance
  representation.
- General context uses up to 7 previous dialogue turns.
- Speaker-state uses up to 8 previous utterances from the same speaker.
- Weighted F1 is used for comparison with prior MELD ERC work.
- Macro F1 is also reported because MELD is class imbalanced.

## Ablation vs Faithfulness

Training-level ablation and explanation faithfulness are different analyses.

- Training-level ablation removes a component before training and measures its
  effect on overall prediction performance.
- Inference-level ablation masks a component in a trained model and measures
  sensitivity at test time.
- Faithfulness analysis checks whether evidence identified as important for an
  individual prediction changes confidence or labels when masked or perturbed.

Use training-level ablation for component contribution claims. Use masking,
token occlusion, and Integrated Gradients for explanation-faithfulness claims.
