from __future__ import annotations

from typing import Any, Dict, List


def normalize_label(sample: Dict[str, Any]) -> str:
    if "label_name" in sample:
        return str(sample["label_name"])
    if "emotion" in sample:
        return str(sample["emotion"])
    if "emotions" in sample:
        return str(sample["emotions"])
    if "label" in sample and isinstance(sample["label"], str):
        return sample["label"]
    raise KeyError("Could not find emotion label in sample")


def build_model_text(sample: Dict[str, Any], include_context: bool = True) -> str:
    """Build text input for ERC: context turns + current turn."""
    current = f"{sample.get('speaker', 'speaker')}: {sample.get('utterance', '')}"
    if not include_context:
        return current
    turns = sample.get("context_turns", []) or []
    if turns:
        context = " </s> ".join([f"{t.get('speaker', 'speaker')}: {t.get('utterance', '')}" for t in turns])
    else:
        context = sample.get("context", "")
    return f"{context} </s> {current}" if context else current


def build_speaker_state_text(sample: Dict[str, Any], include_speaker_history: bool = True) -> str:
    """Build same-speaker history text used for latent speaker-state evolution."""
    if not include_speaker_history:
        return f"{sample.get('speaker', 'speaker')}: {sample.get('utterance', '')}"
    history_turns = sample.get("speaker_history_turns", []) or []
    if history_turns:
        text = " </s> ".join([f"{t.get('speaker', sample.get('speaker', 'speaker'))}: {t.get('utterance', '')}" for t in history_turns])
    else:
        text = sample.get("speaker_history", "")
    current = f"{sample.get('speaker', 'speaker')}: {sample.get('utterance', '')}"
    return f"{text} </s> {current}" if text else current


def build_turn_samples(dialogues: List[Dict[str, Any]], label2id: Dict[str, int], context_window: int, speaker_history_window: int) -> List[Dict[str, Any]]:
    """Convert dialogue-level JSON into turn-level ERC samples.

    Expected dialogue keys: episode/dialogue_id, speakers, utterances or utterances_en, emotions/labels.
    For MASAC, utterances is preferred so original Hinglish/code-mixed text is preserved.
    """
    samples: List[Dict[str, Any]] = []
    for d_idx, dialog in enumerate(dialogues):
        dialogue_id = dialog.get("dialogue_id") or dialog.get("episode") or f"dialogue_{d_idx}"
        speakers = dialog.get("speakers", [])
        utterances = dialog.get("utterances") or dialog.get("utterances_en", [])
        emotions = dialog.get("emotions") or dialog.get("labels", [])
        if not (len(speakers) == len(utterances) == len(emotions)):
            raise ValueError(f"Length mismatch in dialogue {dialogue_id}: speakers={len(speakers)}, utterances={len(utterances)}, labels={len(emotions)}")
        prior_turns: List[Dict[str, Any]] = []
        for i, (speaker, utt, emo) in enumerate(zip(speakers, utterances, emotions)):
            if emo not in label2id:
                continue
            context_turns = prior_turns[-context_window:] if context_window > 0 else []
            same_speaker = [t for t in prior_turns if t["speaker"] == speaker]
            speaker_hist = same_speaker[-speaker_history_window:] if speaker_history_window > 0 else []
            previous_turn = prior_turns[-1] if prior_turns else None
            previous_emotion = previous_turn["label_name"] if previous_turn else None
            previous_speaker = previous_turn["speaker"] if previous_turn else None
            previous_utterance = previous_turn["utterance"] if previous_turn else None
            sample = {
                "dialogue_id": dialogue_id,
                "turn_id": i,
                "speaker": str(speaker),
                "utterance": str(utt),
                "label_name": str(emo),
                "label": int(label2id[emo]),
                "context_turns": [dict(t) for t in context_turns],
                "speaker_history_turns": [dict(t) for t in speaker_hist],
                "context": " </s> ".join([f"{t['speaker']}: {t['utterance']}" for t in context_turns]),
                "speaker_history": " </s> ".join([f"{t['speaker']}: {t['utterance']}" for t in speaker_hist]),
                "previous_speaker": previous_speaker,
                "previous_utterance": previous_utterance,
                "previous_turn_text": f"{previous_speaker}: {previous_utterance}" if previous_turn else "",
                "previous_emotion": previous_emotion,
                "previous_emotion_id": label2id[previous_emotion] if previous_emotion in label2id else -100,
            }
            samples.append(sample)
            prior_turns.append({
                "dialogue_id": dialogue_id,
                "turn_id": i,
                "speaker": str(speaker),
                "utterance": str(utt),
                "label_name": str(emo),
                "label": int(label2id[emo]),
            })
    return samples
