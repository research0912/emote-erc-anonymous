from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from .base import build_turn_samples


class MELDDataset(Dataset):
    """Robust MELD loader.

    Supports common MELD CSV/TXT files with columns such as:
    Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion.

    Also supports TSV files with columns Speaker, Utterance, Emotion, Sentiment where
    a blank line marks the start of the next conversation.
    """
    def __init__(self, path, label2id, context_window=5, speaker_history_window=5):
        self.path = Path(path)
        self.label2id = label2id
        self.context_window = int(context_window)
        self.speaker_history_window = int(speaker_history_window)
        rows = self._read_rows()
        dialogues = self._rows_to_dialogues(rows)
        self.dialogues = dialogues
        self.samples = build_turn_samples(dialogues, label2id, self.context_window, self.speaker_history_window)

    def _read_rows(self):
        with open(self.path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
        nonblank_lines = [line for line in text.splitlines() if line.strip()]
        if not nonblank_lines:
            return []

        # Inspect first non-blank line to detect format
        first_line = nonblank_lines[0]

        # Check if separated by multiple spaces (e.g. 2 or more spaces)
        if "  " in first_line and "\t" not in first_line and "," not in first_line:
            import re
            lines = text.splitlines()
            header = None
            rows = []
            dialogue_idx = 0
            turn_idx = 0
            saw_row_in_dialogue = False

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if saw_row_in_dialogue:
                        dialogue_idx += 1
                        turn_idx = 0
                        saw_row_in_dialogue = False
                    continue

                parts = re.split(r'\s{2,}', stripped)
                if not parts or all(not p.strip() for p in parts):
                    continue

                if header is None:
                    header = [p.strip() for p in parts]
                    continue

                values = parts + [""] * max(0, len(header) - len(parts))
                row = {key: value.strip() for key, value in zip(header, values)}
                row["__dialogue_id"] = f"dialogue_{dialogue_idx}"
                row["__turn_id"] = str(turn_idx)
                rows.append(row)
                turn_idx += 1
                saw_row_in_dialogue = True
            return rows

        # Check if comma-separated
        if "," in first_line and "\t" not in first_line:
            reader = csv.reader(text.splitlines())
        # Otherwise assume tab-separated
        else:
            reader = csv.reader(text.splitlines(), delimiter="\t")

        header = None
        rows = []
        dialogue_idx = 0
        turn_idx = 0
        saw_row_in_dialogue = False

        for raw_row in reader:
            if not raw_row or all(not value.strip() for value in raw_row):
                if saw_row_in_dialogue:
                    dialogue_idx += 1
                    turn_idx = 0
                    saw_row_in_dialogue = False
                continue

            if header is None:
                header = [value.strip() for value in raw_row]
                if len(header) == 1 and "," in header[0]:
                    reader = csv.reader(text.splitlines(), delimiter=",")
                    header = None
                    rows = []
                    dialogue_idx = 0
                    turn_idx = 0
                    saw_row_in_dialogue = False
                continue

            values = raw_row + [""] * max(0, len(header) - len(raw_row))
            row = {key: value.strip() for key, value in zip(header, values)}
            row["__dialogue_id"] = f"dialogue_{dialogue_idx}"
            row["__turn_id"] = str(turn_idx)
            rows.append(row)
            turn_idx += 1
            saw_row_in_dialogue = True

        return rows

    def _get(self, row, names, default=""):
        lower = {k.lower(): k for k in row.keys()}
        for name in names:
            if name in row and row[name] != "":
                return row[name]
            if name.lower() in lower and row[lower[name.lower()]] != "":
                return row[lower[name.lower()]]
        return default

    def _rows_to_dialogues(self, rows):
        grouped = defaultdict(list)
        for idx, row in enumerate(rows):
            did = self._get(row, ["Dialogue_ID", "dialogue_id", "DialogueID", "dialogue", "__dialogue_id"], "dialogue_0")
            turn = self._get(row, ["Utterance_ID", "utterance_id", "UtteranceID", "turn_id", "__turn_id"], str(idx))
            try:
                turn_i = int(turn)
            except Exception:
                turn_i = idx
            grouped[str(did)].append((turn_i, row))
        dialogues = []
        for did, items in grouped.items():
            items = sorted(items, key=lambda x: x[0])
            speakers, utterances, emotions = [], [], []
            for _, row in items:
                speakers.append(self._get(row, ["Speaker", "speaker"], "speaker"))
                utterances.append(self._get(row, ["Utterance", "utterance", "text"], ""))
                emotions.append(self._get(row, ["Emotion", "emotion", "label", "Label"], "neutral"))
            dialogues.append({"dialogue_id": did, "speakers": speakers, "utterances": utterances, "emotions": emotions})
        return dialogues

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
