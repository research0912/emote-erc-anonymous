import argparse
import csv
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.factory import build_dataset
from src.training.utils import get_device, load_checkpoint
from src.models.context_speaker_shift_model import ContextSpeakerShiftERC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_examples", type=int, default=50)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    labels = cfg["dataset"]["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    path = cfg["dataset"][f"{args.split}_path"]
    ds = build_dataset(cfg["dataset"]["name"], path, label2id, cfg["dataset"].get("context_window", 5), cfg["dataset"].get("speaker_history_window", cfg["dataset"].get("context_window", 5)))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    device = get_device()
    model = ContextSpeakerShiftERC(cfg["model"], len(labels)).to(device) if False else ContextSpeakerShiftERC(cfg, len(labels)).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    out_dir = Path(cfg["experiment"]["output_dir"]) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "attention_uncertainty.csv"

    rows = []
    count = 0
    for sample in ds.samples:
        if count >= args.num_examples:
            break
        # build tensors via tokenizer
        # current turn
        utt = sample["utterance"]
        enc = tokenizer(utt, truncation=True, max_length=cfg["dataset"].get("turn_max_length", 96), return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # context turns if present
        context_turns = sample.get("context", [])
        if context_turns:
            ctx_enc = tokenizer(context_turns, truncation=True, padding=True, max_length=cfg["dataset"].get("turn_max_length", 96), return_tensors="pt")
            ctx_ids = ctx_enc["input_ids"].unsqueeze(0).to(device) if ctx_enc["input_ids"].dim()==2 else ctx_enc["input_ids"].to(device)
            # ensure shape [B, num_turns, seq_len]
            if ctx_ids.dim()==2:
                ctx_ids = ctx_ids.unsqueeze(0)
            ctx_mask = ctx_enc["attention_mask"].unsqueeze(0).to(device)
            # create context_turn_mask: all ones
            context_turn_mask = torch.ones((1, ctx_ids.size(1)), dtype=torch.long, device=device)
        else:
            ctx_ids = None
            ctx_mask = None
            context_turn_mask = None

        # MC-dropout prediction
        mean_logits, std_logits = model.mc_dropout_predict(n_samples=model.mc_dropout_samples, input_ids=input_ids, attention_mask=attention_mask, context_input_ids=ctx_ids, context_attention_mask=ctx_mask, context_turn_mask=context_turn_mask, current_turn_indices=torch.tensor([0], device=device))
        probs = torch.softmax(mean_logits, dim=-1)[0].cpu().tolist()
        pred = int(mean_logits.argmax(dim=-1)[0].cpu().item())
        entropy = model.predictive_entropy(torch.tensor(probs)).item()
        std_max = float(std_logits.max(dim=-1)[0].cpu().item())

        # attention weights
        attn_weights = None
        dominant_idx = None
        if model.cross_attention is not None and ctx_ids is not None:
            # compute pooled context repr per turn
            batch_size, num_turns, seq_len = ctx_ids.shape
            flat_ids = ctx_ids.view(batch_size * num_turns, seq_len)
            flat_mask = ctx_mask.view(batch_size * num_turns, seq_len)
            with torch.no_grad():
                context_turn_repr = model._encode_pooled(flat_ids, flat_mask).view(batch_size, num_turns, -1)
                query = model._encode_pooled(input_ids, attention_mask)
                # call cross_attention directly to get weights
                q = query.unsqueeze(1)
                # key_padding_mask expects True for positions that are PAD -> we pass (context_turn_mask==0)
                key_padding_mask = (context_turn_mask == 0) if context_turn_mask is not None else None
                attn_out, weights = model.cross_attention(q, context_turn_repr, context_turn_repr, key_padding_mask=key_padding_mask)
                # weights shape [B, num_heads, query_len, key_len] or [B, query_len, key_len] depending on PyTorch
                # convert to list
                w = weights.cpu().numpy().tolist()
                attn_weights = w
                # compute mean across heads and query_len
                import numpy as np
                arr = np.array(w)
                if arr.ndim == 4:
                    mean_w = arr.mean(axis=(1,2))[0]
                else:
                    mean_w = arr.mean(axis=0)[0]
                dominant_idx = int(mean_w.argmax())

        rows.append({
            "dialogue_id": sample.get("dialogue_id"),
            "turn_id": sample.get("turn_id"),
            "prediction": pred,
            "prediction_confidence": max(probs),
            "entropy": float(entropy),
            "std_max": std_max,
            "attention_weights": json.dumps(attn_weights),
            "dominant_attention_idx": dominant_idx,
        })
        count += 1

    # write csv
    with open(out_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved diagnostics to {out_csv}")


if __name__ == '__main__':
    main()
