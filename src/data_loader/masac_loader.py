from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset
from .base import build_turn_samples


class MASACDataset(Dataset):
    """MASAC loader for dialogue JSON files.

    Supports files where each item is a dialogue with speakers, utterances_en, utterances,
    and emotions/labels. The shared sample builder prefers utterances as the model input text.
    """
    def __init__(self, path, label2id, context_window=5, speaker_history_window=5):
        self.path = Path(path)
        self.label2id = label2id
        self.context_window = int(context_window)
        self.speaker_history_window = int(speaker_history_window)
        with open(self.path, "r", encoding="utf-8") as f:
            dialogues = json.load(f)
        self.dialogues = dialogues
        self.samples = build_turn_samples(dialogues, label2id, self.context_window, self.speaker_history_window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
