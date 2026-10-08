import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(y_true: List[int], y_pred: List[int], id2label: Dict[int, str]):
    labels = list(sorted(id2label.keys()))
    target_names = [id2label[i] for i in labels]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report_dict": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
        ),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(out_dir, model, optimizer, scheduler, epoch, step, best_metric):
    root = Path(out_dir)
    ckpt_dir = root / f"checkpoint-step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
    }, ckpt_dir / "training_state.pt")
    (root / "latest_checkpoint.txt").write_text(str(ckpt_dir), encoding="utf-8")
    for old_ckpt in root.glob("checkpoint-step-*"):
        if old_ckpt != ckpt_dir and old_ckpt.is_dir():
            shutil.rmtree(old_ckpt)
    return ckpt_dir


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    state = torch.load(Path(path) / "training_state.pt", map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    return state
