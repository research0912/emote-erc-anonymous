from .meld_loader import MELDDataset
from .masac_loader import MASACDataset


def build_dataset(name, path, label2id, context_window=5, speaker_history_window=5):
    name = name.lower()
    if name == "meld":
        return MELDDataset(path, label2id, context_window, speaker_history_window)
    if name == "masac":
        return MASACDataset(path, label2id, context_window, speaker_history_window)
    raise ValueError(f"Unsupported dataset: {name}")
