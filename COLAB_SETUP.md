# Running Experiments and Advanced XAI on Google Colab

## Overview
This guide explains how to package this repository, upload it, and run both multi-seed experiments and the **12-module advanced XAI dashboard suite** on Google Colab.

---

## 🚀 Advanced XAI and Selective Colab Export

To simplify deploying this repository to Google Colab, we have added a selective packaging tool and a rich interactive notebook dashboard:

### 1. Selective Project Packaging
Run the custom packaging script locally to create a lightweight zip file, automatically excluding massive system folders (like `.git`, `__pycache__`, or local virtual envs):
```bash
python scripts/export_to_colab.py
```
This produces `erc_colab_export.zip` in your root directory. Download this file or upload it directly to your Google Drive to be mounted in Google Colab.

### 2. Interactive Jupyter Notebook
A complete Jupyter Notebook has been prepared at `Colab_Advanced_XAI.ipynb`. You can open this notebook inside Google Colab to run the advanced explainability suite and render interactive charts (polar radar charts, trajectory line plots, token consistency matrices, and natural language explanation paragraphs).

---

## Colab Setup Steps

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

### Step 2: Navigate to Project and Install Dependencies
```bash
cd /content/drive/MyDrive/path/to/erc-context-speaker-shift-xai
pip install -q torch transformers peft accelerate scikit-learn pyyaml pandas tqdm
```

### Step 3: Verify MELD Data is Accessible
```python
import os
from pathlib import Path

data_dir = Path('/content/drive/MyDrive/path/to/erc-context-speaker-shift-xai/data/meld')
print("MELD files:")
for f in sorted(data_dir.glob('*')):
    print(f"  {f.name}")
```

### Step 4: Run Single Seed Test (Optional)
```bash
cd /content/drive/MyDrive/path/to/erc-context-speaker-shift-xai
python train.py --config configs/meld_roberta_v8_multiseed_base.yaml
```

### Step 5: Run Multi-Seed Experiment
```bash
cd /content/drive/MyDrive/path/to/erc-context-speaker-shift-xai

python scripts/run_multiseed_experiment_colab.py \
  --config /content/drive/MyDrive/path/to/erc-context-speaker-shift-xai/configs/meld_roberta_v8_multiseed_base.yaml \
  --seeds 42,123,456 \
  --output-tag v8_multiseed_colab \
  --project-root /content/drive/MyDrive/path/to/erc-context-speaker-shift-xai
```

### Step 6: Monitor Progress
Each seed will print diagnostics:
```
================================================================================
VARIANT DIAGNOSTICS
================================================================================
Experiment: meld_roberta_v8_multiseed_base
Seed: 42
Active Features:
  use_bilstm:          True
  use_gnn:             False
  use_cross_attention: True
  ...
```

### Step 7: Retrieve Results
After all seeds complete, results will be in:
```
outputs/multiseed_summary_v8_multiseed_colab/
├── aggregated_metrics.json       # Machine-readable stats
└── detailed_breakdown.txt        # Human-readable mean/std/min/max
```

Download from Drive manually or read in Colab:
```python
import json

summary_path = '/content/drive/MyDrive/path/to/erc-context-speaker-shift-xai/outputs/multiseed_summary_v8_multiseed_colab/aggregated_metrics.json'
with open(summary_path) as f:
    results = json.load(f)

for split in ['val', 'test']:
    print(f"\n{split.upper()}:")
    for metric, stats in results[split].items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

---

## Complete Colab Notebook Cells

Copy-paste this sequence into a Colab notebook:

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```

### Cell 2: Set Paths and Install
```python
import os
from pathlib import Path

PROJECT_ROOT = '/content/drive/MyDrive/Research/ERC_ICTAI/erc-context-speaker-shift-xai'
os.chdir(PROJECT_ROOT)

# Install dependencies
os.system('pip install -q torch transformers peft accelerate scikit-learn pyyaml pandas tqdm')
print(f"Working directory: {os.getcwd()}")
```

### Cell 3: Verify Setup
```python
# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Check data
data_dir = Path('data/meld')
print(f"\nMELD data files:")
for f in sorted(data_dir.glob('MELD_*')):
    print(f"  {f.name}")

# Check configs
config_file = Path('configs/meld_roberta_v8_multiseed_base.yaml')
print(f"\nConfig exists: {config_file.exists()}")
```

### Cell 4: Run Multi-Seed (3 seeds = ~2-3 hours)
```python
import subprocess
import sys

result = subprocess.run([
    sys.executable, 'scripts/run_multiseed_experiment_colab.py',
    '--config', 'configs/meld_roberta_v8_multiseed_base.yaml',
    '--seeds', '42,123,456',
    '--output-tag', 'v8_multiseed_colab',
    '--project-root', PROJECT_ROOT
], cwd=PROJECT_ROOT)

print(f"\nExperiment exit code: {result.returncode}")
```

### Cell 5: Load and Display Results
```python
import json
from pathlib import Path

summary_dir = Path('outputs/multiseed_summary_v8_multiseed_colab')

# Load aggregated metrics
metrics_file = summary_dir / 'aggregated_metrics.json'
with open(metrics_file) as f:
    agg = json.load(f)

print("="*80)
print("MULTI-SEED RESULTS")
print("="*80)

for split in ['val', 'test']:
    print(f"\n{split.upper()}:")
    print("-"*60)
    for metric, stats in sorted(agg[split].items()):
        print(f"  {metric:30s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

# Load detailed breakdown
with open(summary_dir / 'detailed_breakdown.txt') as f:
    print("\n" + f.read())
```

---

## Key Differences from Local

| Aspect | Local | Colab |
|--------|-------|-------|
| **Path to data** | `data/meld/` | `/content/drive/MyDrive/.../erc-context-speaker-shift-xai/data/meld/` |
| **Script to use** | `run_multiseed_experiment.py` | `run_multiseed_experiment_colab.py` |
| **Project root arg** | Not needed | `--project-root /content/drive/MyDrive/.../erc-context-speaker-shift-xai` |
| **GPU** | NVIDIA (local) | TPU/T4/A100 (Colab) |
| **Persistence** | On disk | In Drive (survives session end) |

---

## Time Estimates

- **3 seeds** (42, 123, 456): ~2–3 hours (Colab T4 GPU)
- **5 seeds** (+ 789, 999): ~4–5 hours

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'peft'`
```python
os.system('pip install -U peft')
```

### `CUDA out of memory`
Reduce `batch_size` in the config from 8 → 4, or reduce `epochs` from 12 → 10.

### Data not found
Verify path to MELD files:
```python
from pathlib import Path
p = Path('/content/drive/MyDrive/Research/ERC_ICTAI/erc-context-speaker-shift-xai/data/meld')
print(list(p.glob('MELD_*')))
```

### Session timeout
Results are saved to Drive in `outputs/multiseed_summary_v8_multiseed_colab/` — re-run cell 5 to load existing results without re-running training.

---

## Sample Output

After Cell 4 completes, you'll see:

```
================================================================================
MULTISEED SUMMARY
================================================================================

VAL METRICS (mean ± std):
------------------------------------------------------------
  accuracy                      : 0.6540 ± 0.0085  [min=0.6412, max=0.6681]
  macro_f1                      : 0.4623 ± 0.0156  [min=0.4421, max=0.4802]
  micro_f1                      : 0.6540 ± 0.0085  [min=0.6412, max=0.6681]
  weighted_f1                   : 0.6341 ± 0.0092  [min=0.6201, max=0.6521]

TEST METRICS (mean ± std):
------------------------------------------------------------
  accuracy                      : 0.6589 ± 0.0112  [min=0.6401, max=0.6756]
  macro_f1                      : 0.4701 ± 0.0189  [min=0.4456, max=0.4921]
  micro_f1                      : 0.6589 ± 0.0112  [min=0.6401, max=0.6756]
  weighted_f1                   : 0.6412 ± 0.0101  [min=0.6251, max=0.6623]

Results saved to: outputs/multiseed_summary_v8_multiseed_colab
  - aggregated_metrics.json
  - detailed_breakdown.txt
```

The `± std` values tell you whether the variant is stable across different random seeds.
