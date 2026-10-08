from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _save_transition_outputs(samples, out):
    rows = []
    for sample in samples:
        current = sample.get("label_name")
        if current is None:
            continue
        rows.append({
            "previous_emotion": sample.get("previous_emotion") or "START",
            "current_emotion": current,
        })

    if not rows:
        return

    transition_df = pd.DataFrame(rows)
    matrix = pd.crosstab(transition_df["previous_emotion"], transition_df["current_emotion"])
    matrix.to_csv(out / "emotion_transition_matrix.csv")

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0)
    plt.colorbar(label="Count")
    plt.title("Emotion Transition Heatmap")
    plt.xlabel("Current Emotion")
    plt.ylabel("Previous Emotion")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.tight_layout()
    plt.savefig(out / "emotion_transition_heatmap.png", dpi=200)
    plt.close()


def dataset_distribution(dataset, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [s["label_name"] for s in dataset.samples]
    speakers = [s["speaker"] for s in dataset.samples]
    context_lengths = [len(s.get("context_turns", [])) for s in dataset.samples]
    speaker_history_lengths = [len(s.get("speaker_history_turns", [])) for s in dataset.samples]

    label_df = pd.DataFrame(Counter(labels).items(), columns=["label", "count"]).sort_values("label")
    speaker_df = pd.DataFrame(Counter(speakers).most_common(30), columns=["speaker", "count"])
    stats = pd.DataFrame({
        "context_length": context_lengths,
        "speaker_history_length": speaker_history_lengths,
    })
    label_df.to_csv(out / "label_distribution.csv", index=False)
    speaker_df.to_csv(out / "top_speaker_distribution.csv", index=False)
    stats.describe().to_csv(out / "context_history_length_stats.csv")

    plt.figure(figsize=(8, 4))
    plt.bar(label_df["label"], label_df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Emotion Label Distribution")
    plt.tight_layout()
    plt.savefig(out / "label_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(context_lengths, bins=20)
    plt.title("Context Window Length Distribution")
    plt.xlabel("Number of context turns")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "context_length_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(speaker_history_lengths, bins=20)
    plt.title("Speaker-History Length Distribution")
    plt.xlabel("Number of same-speaker history turns")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out / "speaker_history_length_distribution.png", dpi=200)
    plt.close()

    _save_transition_outputs(dataset.samples, out)
