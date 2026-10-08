import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv(csv_path):
    return pd.read_csv(csv_path)


def _save_current(out_path):
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_confusion(metrics_path, out_path):
    m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    cm = m["confusion_matrix"]
    labels = list(m["classification_report_dict"].keys())
    labels = [x for x in labels if x not in ["accuracy", "macro avg", "weighted avg"]]
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0)
    plt.title("Confusion Matrix")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            plt.text(j, i, str(val), ha="center", va="center", fontsize=8)
    _save_current(out_path)


def plot_ablation(csv_path, out_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(7, 4))
    plt.bar(df["setting"], df["macro_f1"])
    plt.ylabel("Macro F1")
    plt.title("Ablation Study")
    _save_current(out_path)


def plot_emotion_distribution(csv_path, out_path):
    df = _read_csv(csv_path)
    label_col = _first_existing_column(df, ["label", "emotion", "gold", "class"])
    count_col = _first_existing_column(df, ["count", "support", "n", "frequency"])
    if label_col is None or count_col is None:
        raise ValueError(f"{csv_path} must include label/count columns")

    df = df.sort_values(count_col, ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(df[label_col].astype(str), df[count_col])
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.title("Emotion Distribution")
    plt.xticks(rotation=45, ha="right")
    _save_current(out_path)


def plot_transition_heatmap(csv_path, out_path):
    df = _read_csv(csv_path)
    prev_col = _first_existing_column(df, ["previous_emotion", "previous", "prev_emotion"])
    cur_col = _first_existing_column(df, ["current_emotion", "current", "gold", "label"])

    if prev_col and cur_col:
        matrix = pd.crosstab(df[prev_col].fillna("START"), df[cur_col])
    else:
        first_col = df.columns[0]
        if first_col.startswith("Unnamed"):
            matrix = df.set_index(first_col)
        else:
            matrix = df.set_index(first_col) if first_col not in df.select_dtypes("number").columns else df
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0)
    plt.colorbar(label="Count")
    plt.title("Emotion Transition Heatmap")
    plt.xlabel("Current Emotion")
    plt.ylabel("Previous Emotion")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    _save_current(out_path)


def plot_xai_confidence_drop(csv_path, out_path):
    df = _read_csv(csv_path)
    drop_cols = [
        col for col in ["top_context_drop", "speaker_state_drop", "emotion_shift_drop", "shift_drop"]
        if col in df.columns
    ]
    if not drop_cols:
        raise ValueError(f"{csv_path} must include at least one XAI drop column")

    means = df[drop_cols].mean().sort_values(ascending=False)
    labels = [col.replace("_drop", "").replace("_", " ").title() for col in means.index]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means.values)
    plt.ylabel("Mean Gold-Probability Drop")
    plt.title("XAI Confidence Drop by Evidence Source")
    plt.xticks(rotation=25, ha="right")
    _save_current(out_path)


def plot_context_importance(csv_path, out_path):
    df = _read_csv(csv_path)
    index_col = _first_existing_column(df, ["masked_context_index", "context_index", "turn_offset"])
    drop_col = _first_existing_column(df, ["gold_probability_drop", "confidence_drop"])
    if index_col is None or drop_col is None:
        raise ValueError(f"{csv_path} must include context-index/drop columns")

    grouped = df.groupby(index_col, dropna=False)[drop_col].mean().reset_index()
    grouped = grouped.sort_values(index_col)
    plt.figure(figsize=(7, 4))
    plt.bar(grouped[index_col].astype(str), grouped[drop_col])
    plt.xlabel("Masked Context Turn Index")
    plt.ylabel("Mean Gold-Probability Drop")
    plt.title("Context Turn Importance")
    _save_current(out_path)


def plot_speaker_state_drop(csv_path, out_path):
    df = _read_csv(csv_path)
    drop_col = _first_existing_column(df, ["gold_probability_drop", "speaker_state_drop", "confidence_drop"])
    if drop_col is None:
        raise ValueError(f"{csv_path} must include a speaker-state drop column")

    plt.figure(figsize=(7, 4))
    plt.hist(df[drop_col].dropna(), bins=20)
    plt.xlabel("Gold-Probability Drop")
    plt.ylabel("Examples")
    plt.title("Speaker-State Masking Drop")
    _save_current(out_path)


def plot_encoder_comparison(csv_path, out_path):
    df = _read_csv(csv_path)
    name_col = _first_existing_column(df, ["encoder", "model", "setting", "config", "experiment"])
    metric_cols = [col for col in ["accuracy", "macro_f1", "weighted_f1"] if col in df.columns]
    if name_col is None or not metric_cols:
        raise ValueError(f"{csv_path} must include an encoder/model column and at least one metric column")

    plot_df = df[[name_col] + metric_cols].copy()
    plot_df = plot_df.set_index(name_col)
    ax = plot_df.plot(kind="bar", figsize=(8, 4.5))
    ax.set_xlabel("Encoder")
    ax.set_ylabel("Score")
    ax.set_title("Encoder Comparison")
    ax.legend(title="Metric")
    plt.xticks(rotation=30, ha="right")
    _save_current(out_path)


def plot_explanation_stability(csv_paths, out_csv_path, out_png_path):
    frames = []
    for csv_path in csv_paths:
        path = Path(csv_path)
        df = _read_csv(path)
        key_cols = [col for col in ["dialogue_id", "turn_id"] if col in df.columns]
        drop_cols = [
            col for col in ["top_context_drop", "speaker_state_drop", "emotion_shift_drop", "shift_drop"]
            if col in df.columns
        ]
        if len(key_cols) < 2 or not drop_cols:
            raise ValueError(f"{path} must include dialogue_id, turn_id, and XAI drop columns")
        run_id = path.parent.name if path.parent.name else path.stem
        frame = df[key_cols + drop_cols].copy()
        frame["run_id"] = run_id
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    drop_cols = [
        col for col in ["top_context_drop", "speaker_state_drop", "emotion_shift_drop", "shift_drop"]
        if col in merged.columns
    ]
    rows = []
    for source in drop_cols:
        per_example = merged.groupby(["dialogue_id", "turn_id"])[source].agg(["mean", "std", "count"]).reset_index()
        per_example = per_example[per_example["count"] > 1]
        if per_example.empty:
            continue
        rows.append({
            "source": source,
            "num_examples": int(len(per_example)),
            "mean_drop": float(per_example["mean"].mean()),
            "mean_std": float(per_example["std"].fillna(0).mean()),
            "stability_score": float(1.0 / (1.0 + per_example["std"].fillna(0).mean())),
        })

    if not rows:
        raise ValueError("Need at least two XAI summaries with overlapping dialogue_id/turn_id values")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv_path, index=False)

    labels = summary["source"].str.replace("_drop", "", regex=False).str.replace("_", " ").str.title()
    plt.figure(figsize=(7, 4))
    plt.bar(labels, summary["stability_score"])
    plt.ylim(0, 1.05)
    plt.ylabel("Stability Score")
    plt.title("Explanation Stability Across Runs")
    plt.xticks(rotation=25, ha="right")
    _save_current(out_png_path)


def _maybe_plot(label, csv_path, out_path, plotter):
    if not csv_path:
        return
    path = Path(csv_path)
    if not path.exists():
        print(f"Skipping {label}: missing {path}")
        return
    plotter(path, out_path)
    print(f"Saved {out_path}")


def _infer_transition_csv(args):
    if args.transition_csv:
        return args.transition_csv
    if args.xai_summary_csv:
        candidate = Path(args.xai_summary_csv).parent / "emotion_transition_matrix.csv"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", help="Path to metrics JSON")
    parser.add_argument("--ablation_csv", help="Path to ablation summary CSV")
    parser.add_argument("--label_distribution_csv", help="Path to label_distribution.csv")
    parser.add_argument("--xai_summary_csv", help="Path to xai/xai_summary.csv")
    parser.add_argument("--context_importance_csv", help="Path to xai/context_turn_importance.csv")
    parser.add_argument("--speaker_state_csv", help="Path to xai/speaker_state_importance.csv")
    parser.add_argument("--encoder_comparison_csv", help="Path to encoder_comparison.csv")
    parser.add_argument("--transition_csv", help="Optional path to emotion transition matrix or transition rows CSV")
    parser.add_argument("--stability_xai_summary_csvs", nargs="+", help="Two or more xai_summary.csv files from different runs/seeds")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.metrics:
        plot_confusion(args.metrics, out / "confusion_matrix.png")
    if args.ablation_csv:
        plot_ablation(args.ablation_csv, out / "ablation_macro_f1.png")
    _maybe_plot("emotion distribution", args.label_distribution_csv, out / "emotion_distribution.png", plot_emotion_distribution)
    _maybe_plot("XAI confidence drop", args.xai_summary_csv, out / "xai_confidence_drop.png", plot_xai_confidence_drop)
    _maybe_plot("context importance", args.context_importance_csv, out / "context_importance.png", plot_context_importance)
    _maybe_plot("speaker-state drop", args.speaker_state_csv, out / "speaker_state_drop.png", plot_speaker_state_drop)
    _maybe_plot("encoder comparison", args.encoder_comparison_csv, out / "encoder_comparison.png", plot_encoder_comparison)

    transition_csv = _infer_transition_csv(args)
    _maybe_plot("transition heatmap", transition_csv, out / "transition_heatmap.png", plot_transition_heatmap)
    if args.stability_xai_summary_csvs:
        plot_explanation_stability(
            args.stability_xai_summary_csvs,
            out / "explanation_stability.csv",
            out / "explanation_stability.png",
        )
        print(f"Saved {out / 'explanation_stability.csv'}")
        print(f"Saved {out / 'explanation_stability.png'}")


if __name__ == "__main__":
    main()
