import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data_loader.factory import build_dataset
from src.models.context_speaker_shift_model import ContextSpeakerShiftERC
from src.training.collate import build_token_cache_for_dataset, make_collate_fn
from src.training.utils import compute_metrics, get_device, load_checkpoint, save_checkpoint, save_json, set_seed


def compute_class_weights(dataset, num_labels):
    counts = np.zeros(num_labels, dtype=np.float32)
    for s in dataset.samples:
        counts[s["label"]] += 1
    counts[counts == 0] = 1
    return torch.tensor(counts.sum() / (num_labels * counts), dtype=torch.float32)


def compute_effective_num_weights(dataset, num_labels, beta=0.9999):
    counts = np.zeros(num_labels, dtype=np.float32)
    for s in dataset.samples:
        counts[s["label"]] += 1
    counts[counts == 0] = 1
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * num_labels
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, labels):
        ce = torch.nn.functional.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class ClassBalancedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, logits, labels):
        ce = torch.nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class RefinedLabelCrossEntropy(torch.nn.Module):
    def __init__(self, labels, refinement_alpha=0.18, weight=None):
        super().__init__()
        self.labels = list(labels)
        self.refinement_alpha = float(refinement_alpha)
        self.weight = weight
        self.register_buffer("similarity", self._build_similarity(self.labels), persistent=False)

    def _build_similarity(self, labels):
        groups = {
            "anger": {"disgust": 0.45, "sadness": 0.25, "fear": 0.20},
            "disgust": {"anger": 0.45, "sadness": 0.25, "fear": 0.20},
            "fear": {"sadness": 0.35, "surprise": 0.25, "anger": 0.20, "disgust": 0.15},
            "joy": {"surprise": 0.35, "neutral": 0.20},
            "neutral": {"joy": 0.15, "sadness": 0.15, "surprise": 0.10},
            "sadness": {"fear": 0.35, "anger": 0.20, "disgust": 0.20, "neutral": 0.15},
            "surprise": {"joy": 0.35, "fear": 0.25, "neutral": 0.15},
        }
        num_labels = len(labels)
        matrix = torch.zeros(num_labels, num_labels, dtype=torch.float32)
        for i, label in enumerate(labels):
            related = groups.get(label, {})
            for j, other in enumerate(labels):
                if other != label:
                    matrix[i, j] = related.get(other, 0.0)
            total = matrix[i].sum()
            if total > 0:
                matrix[i] = matrix[i] / total
        return matrix

    def forward(self, logits, labels):
        num_labels = logits.size(-1)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=num_labels).float()
        soft_neighbors = self.similarity.to(logits.device)[labels]
        targets = (1.0 - self.refinement_alpha) * one_hot + self.refinement_alpha * soft_neighbors
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        per_class_loss = -targets * log_probs
        if self.weight is not None:
            per_class_loss = per_class_loss * self.weight.to(logits.device).view(1, -1)
        return per_class_loss.sum(dim=-1).mean()


class WeightedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        targets = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        return loss.mean()


def make_loss_fn(cfg, train_ds, num_labels, device):
    training_cfg = cfg["training"]
    loss_type = training_cfg.get("loss_type", "cross_entropy")
    class_weights = None

    if loss_type in {"class_balanced_focal", "cb_focal", "weighted_ce", "focal", "cross_entropy", "weighted_bce", "refined_ce"}:
        if loss_type in {"class_balanced_focal", "cb_focal"}:
            class_weights = compute_effective_num_weights(
                train_ds,
                num_labels,
                beta=training_cfg.get("class_balanced_beta", 0.9999),
            ).to(device)
        elif training_cfg.get("class_weighted_loss", True):
            class_weights = compute_class_weights(train_ds, num_labels).to(device)

    if loss_type in {"class_balanced_focal", "cb_focal"}:
        return ClassBalancedFocalLoss(
            gamma=training_cfg.get("focal_gamma", 1.5),
            label_smoothing=training_cfg.get("label_smoothing", 0.0),
            weight=class_weights,
        )
    if loss_type == "focal":
        return FocalLoss(gamma=training_cfg.get("focal_gamma", 2.0), weight=class_weights)
    if loss_type == "weighted_bce":
        return WeightedBinaryCrossEntropy(pos_weight=class_weights)
    if loss_type == "weighted_ce":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "refined_ce":
        return RefinedLabelCrossEntropy(
            cfg["dataset"]["labels"],
            refinement_alpha=training_cfg.get("refinement_alpha", 0.18),
            weight=class_weights,
        )
    return torch.nn.CrossEntropyLoss(weight=class_weights)


def compute_contrastive_pair_loss(model, view_a, view_b):
    if not getattr(model, "use_contrastive", False):
        return None
    return model.compute_contrastive_loss(view_a, view_b)


MODEL_INPUT_KEYS = [
    "input_ids", "attention_mask",
    "speaker_input_ids", "speaker_attention_mask",
    "previous_input_ids", "previous_attention_mask", "previous_turn_exists",
    "context_input_ids", "context_attention_mask", "context_turn_mask", "context_turn_speaker_ids", "current_turn_indices",
    "previous_emotion_ids",
]


def _resolve_path(path_value, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _checkpoint_has_state(path_value) -> bool:
    return (Path(path_value) / "training_state.pt").exists()


def evaluate(model, loader, device, id2label, loss_fn=None, ablation=None):
    model.eval()
    y_true, y_pred, rows, losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in MODEL_INPUT_KEYS}
            labels = batch["labels"].to(device)
            logits = model(**inputs, ablation=ablation)["logits"]
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, labels).item()))
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            for meta, pred_id, prob_vec in zip(batch["meta"], preds.cpu().tolist(), probs.cpu().tolist()):
                rows.append({
                    "dialogue_id": meta["dialogue_id"],
                    "turn_id": meta["turn_id"],
                    "speaker": meta["speaker"],
                    "utterance": meta["utterance"],
                    "gold": meta["label_name"],
                    "pred": id2label[pred_id],
                    "confidence": float(max(prob_vec)),
                    "previous_emotion": meta.get("previous_emotion"),
                    "context": meta.get("context", ""),
                    "speaker_history": meta.get("speaker_history", ""),
                })
    metrics = compute_metrics(y_true, y_pred, id2label)
    metrics["loss"] = float(np.mean(losses)) if losses else None
    metrics["predictions"] = rows
    model.train()
    return metrics


def make_loaders(cfg, tokenizer):
    labels = cfg["dataset"]["labels"]
    label2id = {label: i for i, label in enumerate(labels)}
    train_ds = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["train_path"], label2id,
        cfg["dataset"]["context_window"], cfg["dataset"].get("speaker_history_window", cfg["dataset"]["context_window"])
    )
    val_ds = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["val_path"], label2id,
        cfg["dataset"]["context_window"], cfg["dataset"].get("speaker_history_window", cfg["dataset"]["context_window"])
    )
    test_ds = build_dataset(
        cfg["dataset"]["name"], cfg["dataset"]["test_path"], label2id,
        cfg["dataset"]["context_window"], cfg["dataset"].get("speaker_history_window", cfg["dataset"]["context_window"])
    )
    loader_cfg = cfg.get("data_loader", {})
    use_token_cache = bool(loader_cfg.get("cache_tokenized", False))
    if use_token_cache:
        cache_kwargs = {
            "max_length": cfg["dataset"]["max_length"],
            "dialogue_flow_source": cfg["model"].get("dialogue_flow_source", "token"),
            "turn_max_length": cfg["dataset"].get("turn_max_length", 96),
            "main_input_include_context_when_turn_level": cfg["model"].get("main_input_include_context_when_turn_level", False),
            "main_input_include_context": cfg["model"].get("main_input_include_context", True),
            "mask_current_utterance": cfg["model"].get("mask_current_utterance", False),
        }
        for dataset in (train_ds, val_ds, test_ds):
            build_token_cache_for_dataset(dataset, tokenizer, **cache_kwargs)

    collate_fn = make_collate_fn(
        tokenizer,
        cfg["dataset"]["max_length"],
        use_gold_previous_emotion=cfg["dataset"].get("use_gold_previous_emotion", False),
        dialogue_flow_source=cfg["model"].get("dialogue_flow_source", "token"),
        turn_max_length=cfg["dataset"].get("turn_max_length", 96),
        main_input_include_context_when_turn_level=cfg["model"].get("main_input_include_context_when_turn_level", False),
        main_input_include_context=cfg["model"].get("main_input_include_context", True),
        mask_current_utterance=cfg["model"].get("mask_current_utterance", False),
        use_token_cache=use_token_cache,
    )
    num_workers = int(loader_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": cfg["training"]["batch_size"],
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(loader_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(loader_cfg.get("prefetch_factor", 2))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint folder or 'latest'")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    cfg["dataset"]["train_path"] = str(_resolve_path(cfg["dataset"]["train_path"], project_root))
    cfg["dataset"]["val_path"] = str(_resolve_path(cfg["dataset"]["val_path"], project_root))
    cfg["dataset"]["test_path"] = str(_resolve_path(cfg["dataset"]["test_path"], project_root))
    cfg["experiment"]["output_dir"] = str(_resolve_path(cfg["experiment"]["output_dir"], project_root))
    set_seed(cfg["experiment"].get("seed", 42))
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    labels = cfg["dataset"]["labels"]
    id2label = {i: label for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], use_fast=True)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(cfg, tokenizer)
    print(f"Loaded train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    device = get_device()
    model = ContextSpeakerShiftERC(cfg, len(labels)).to(device)
    
    # Print variant diagnostics
    model_cfg = cfg["model"]
    print(f"\n{'='*80}")
    print(f"VARIANT DIAGNOSTICS")
    print(f"{'='*80}")
    print(f"Experiment: {cfg['experiment']['name']}")
    print(f"Seed: {cfg['experiment']['seed']}")
    print(f"\nActive Features:")
    print(f"  use_bilstm:          {model.use_bilstm}")
    print(f"  use_gnn:             {model.use_gnn}")
    print(f"  use_cross_attention: {model.use_cross_attention}")
    print(f"  use_speaker_state:   {model.use_speaker_state}")
    print(f"  use_emotion_shift:   {model.use_emotion_shift}")
    print(f"  use_dynamic_fusion:  {model.use_dynamic_fusion}")
    print(f"  use_mc_dropout:      {model.use_mc_dropout}")
    print(f"  use_contrastive:     {model.use_contrastive}")
    print(f"  use_lora:            {model_cfg.get('use_lora', True)}")
    print(f"\nLoss Configuration:")
    print(f"  loss_type:           {cfg['training'].get('loss_type', 'cross_entropy')}")
    print(f"  focal_gamma:         {cfg['training'].get('focal_gamma', 2.0)}")
    print(f"  class_weighted_loss: {cfg['training'].get('class_weighted_loss', True)}")
    print(f"  contrastive_weight:  {cfg['training'].get('contrastive_weight', 0.0)}")
    print(f"\nTraining Configuration:")
    print(f"  learning_rate:       {cfg['training']['learning_rate']}")
    print(f"  head_learning_rate:  {cfg['training'].get('head_learning_rate', cfg['training']['learning_rate'])}")
    print(f"  epochs:              {cfg['training']['epochs']}")
    print(f"  batch_size:          {cfg['training']['batch_size']}")
    print(f"  early_stopping:      {cfg['training'].get('early_stopping_patience', 4)}")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"{'='*80}\n")
    
    if hasattr(model.encoder, "print_trainable_parameters"):
        model.encoder.print_trainable_parameters()

    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            head_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["training"]["learning_rate"]},
        {"params": head_params, "lr": cfg["training"].get("head_learning_rate", cfg["training"]["learning_rate"])}
    ], weight_decay=cfg["training"].get("weight_decay", 0.01))

    total_steps = math.ceil(len(train_loader) / cfg["training"].get("grad_accum_steps", 1)) * cfg["training"]["epochs"]
    warmup_steps = int(total_steps * cfg["training"].get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = make_loss_fn(cfg, train_ds, len(labels), device)

    use_fp16 = cfg["training"].get("fp16", True) and device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        autocast_ctx = torch.amp.autocast("cuda", enabled=use_fp16)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16)
    start_epoch, global_step, best_metric, patience = 0, 0, -1.0, 0

    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "latest":
            ckpt_path = (out_dir / "latest_checkpoint.txt").read_text(encoding="utf-8").strip()
        state = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("step", 0))
        best_metric = float(state.get("best_metric", -1.0))
        print(f"Resumed from {ckpt_path}")

    model.train()
    grad_accum = cfg["training"].get("grad_accum_steps", 1)
    monitor = cfg["training"].get("monitor_metric", "macro_f1")
    contrastive_weight = cfg["training"].get("contrastive_weight", 0.0)

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", mininterval=10, leave=True, dynamic_ncols=True)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in batch.items() if k in MODEL_INPUT_KEYS}
            y = batch["labels"].to(device)
            with autocast_ctx:
                out_a = model(**inputs)
                logits = out_a["logits"]
                loss = loss_fn(logits, y)
                if contrastive_weight > 0.0 and getattr(model, "use_contrastive", False):
                    out_b = model(**inputs)
                    contrastive_loss = compute_contrastive_pair_loss(model, out_a["features"], out_b["features"])
                    if contrastive_loss is not None:
                        loss = loss + contrastive_weight * contrastive_loss
                loss = loss / grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(loss=float(loss.item() * grad_accum))

                if global_step % cfg["training"].get("save_every_steps", 2000) == 0:
                    save_checkpoint(out_dir, model, optimizer, scheduler, epoch, global_step, best_metric)

                if global_step % cfg["training"].get("eval_every_steps", 2000) == 0:
                    val_metrics = evaluate(model, val_loader, device, id2label, loss_fn)
                    val_metrics_no_preds = {k: v for k, v in val_metrics.items() if k != "predictions"}
                    save_json(val_metrics_no_preds, out_dir / f"val_metrics_step_{global_step}.json")
                    current = val_metrics[monitor]
                    if current > best_metric:
                        best_metric = current
                        patience = 0
                        best_root = out_dir / "best_model"
                        best_root.mkdir(exist_ok=True)
                        save_checkpoint(best_root, model, optimizer, scheduler, epoch, global_step, best_metric)
                    else:
                        patience += 1
                        if patience >= cfg["training"].get("early_stopping_patience", 4):
                            print("Early stopping triggered.")
                            break
        if patience >= cfg["training"].get("early_stopping_patience", 4):
            break

    best_latest = out_dir / "best_model" / "latest_checkpoint.txt"
    if best_latest.exists():
        best_ckpt = best_latest.read_text(encoding="utf-8").strip()
        if _checkpoint_has_state(best_ckpt):
            load_checkpoint(best_ckpt, model, device=device)
        else:
            fallback_latest = out_dir / "latest_checkpoint.txt"
            if fallback_latest.exists():
                fallback_ckpt = fallback_latest.read_text(encoding="utf-8").strip()
                if _checkpoint_has_state(fallback_ckpt):
                    print(f"Best checkpoint is missing; loading latest checkpoint instead: {fallback_ckpt}")
                    load_checkpoint(fallback_ckpt, model, device=device)
                else:
                    print("Best and latest checkpoints are missing; evaluating current in-memory model.")
            else:
                print("Best checkpoint is missing; evaluating current in-memory model.")

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        metrics = evaluate(model, loader, device, id2label, loss_fn)
        preds = metrics.pop("predictions")
        save_json(metrics, out_dir / f"{split_name}_metrics.json")
        import pandas as pd
        pd.DataFrame(preds).to_csv(out_dir / f"{split_name}_predictions.csv", index=False)
        print(f"\n{split_name.upper()} REPORT\n{metrics['classification_report_text']}")


if __name__ == "__main__":
    main()
