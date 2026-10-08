import torch
from src.data_loader.base import build_model_text, build_speaker_state_text


def _current_text(sample, utterance_override=None):
    utterance = sample.get("utterance", "") if utterance_override is None else utterance_override
    return f"{sample.get('speaker', 'speaker')}: {utterance}"


def _turn_texts(sample, include_context=True, current_utterance_override=None):
    context_turns = sample.get("context_turns", []) if include_context else []
    texts = [
        f"{turn.get('speaker', 'speaker')}: {turn.get('utterance', '')}"
        for turn in context_turns
    ]
    texts.append(_current_text(sample, utterance_override=current_utterance_override))
    return texts


def _non_empty_placeholder(tokenizer):
    return tokenizer.unk_token or tokenizer.pad_token or "[NO_TEXT]"


def _speaker_ids_for_turns(sample, include_context=True):
    context_turns = sample.get("context_turns", []) if include_context else []
    speakers = [turn.get("speaker", "<unk>") for turn in context_turns]
    speakers.append(sample.get("speaker", "<unk>"))
    speaker_to_id = {}
    ids = []
    for speaker in speakers:
        if speaker not in speaker_to_id:
            speaker_to_id[speaker] = len(speaker_to_id)
        ids.append(speaker_to_id[speaker])
    return ids


def _pad_sequences(sequences, pad_token_id):
    max_len = max(len(seq) for seq in sequences)
    input_ids, attention_mask = [], []
    for seq in sequences:
        pad_len = max_len - len(seq)
        input_ids.append(seq + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(seq) + [0] * pad_len)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)


def _tokenize_to_ids(tokenizer, texts, max_length):
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    return encoded["input_ids"]


def build_token_cache_for_dataset(
    dataset,
    tokenizer,
    max_length: int,
    dialogue_flow_source: str = "token",
    turn_max_length: int = 96,
    main_input_include_context_when_turn_level: bool = False,
    main_input_include_context: bool = True,
    mask_current_utterance: bool = False,
):
    """Pre-tokenize static dataset fields while preserving dynamic batch padding."""
    use_turn_level_flow = dialogue_flow_source == "turn_level"
    placeholder = _non_empty_placeholder(tokenizer)
    current_override = placeholder if mask_current_utterance else None

    texts, speaker_texts, previous_texts = [], [], []
    turns_by_sample = []
    for sample in dataset.samples:
        text = (
            _current_text(sample, utterance_override=current_override)
            if not main_input_include_context or (use_turn_level_flow and not main_input_include_context_when_turn_level)
            else build_model_text(
                {**sample, "utterance": current_override}
                if current_override is not None else sample,
                include_context=True,
            )
        )
        speaker_text = build_speaker_state_text(
            {**sample, "utterance": current_override}
            if current_override is not None else sample,
            include_speaker_history=True,
        )
        previous_text = sample.get("previous_turn_text") or placeholder
        texts.append(text)
        speaker_texts.append(speaker_text)
        previous_texts.append(previous_text)
        if use_turn_level_flow:
            turns_by_sample.append(_turn_texts(sample, include_context=True, current_utterance_override=current_override))

    main_ids = _tokenize_to_ids(tokenizer, texts, max_length)
    speaker_ids = _tokenize_to_ids(tokenizer, speaker_texts, max_length)
    previous_ids = _tokenize_to_ids(tokenizer, previous_texts, max_length)
    turn_ids = None
    if use_turn_level_flow:
        flat_turns = [turn for turns in turns_by_sample for turn in turns]
        flat_turn_ids = _tokenize_to_ids(tokenizer, flat_turns, turn_max_length)
        turn_ids = []
        cursor = 0
        for turns in turns_by_sample:
            next_cursor = cursor + len(turns)
            turn_ids.append(flat_turn_ids[cursor:next_cursor])
            cursor = next_cursor

    for idx, sample in enumerate(dataset.samples):
        sample["_token_cache"] = {
            "input_ids": main_ids[idx],
            "speaker_input_ids": speaker_ids[idx],
            "previous_input_ids": previous_ids[idx],
        }
        if turn_ids is not None:
            sample["_token_cache"]["context_input_ids"] = turn_ids[idx]


def make_collate_fn(
    tokenizer,
    max_length: int,
    use_gold_previous_emotion: bool = False,
    dialogue_flow_source: str = "token",
    turn_max_length: int = 96,
    main_input_include_context_when_turn_level: bool = False,
    main_input_include_context: bool = True,
    mask_current_utterance: bool = False,
    use_token_cache: bool = False,
):
    def collate(batch):
        use_turn_level_flow = dialogue_flow_source == "turn_level"
        placeholder = _non_empty_placeholder(tokenizer)
        current_override = placeholder if mask_current_utterance else None

        if use_token_cache:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
            enc_ids, enc_mask = _pad_sequences([sample["_token_cache"]["input_ids"] for sample in batch], pad_token_id)
            spk_ids, spk_mask = _pad_sequences([sample["_token_cache"]["speaker_input_ids"] for sample in batch], pad_token_id)
            prev_ids, prev_mask = _pad_sequences([sample["_token_cache"]["previous_input_ids"] for sample in batch], pad_token_id)
        else:
            texts = [
                _current_text(sample, utterance_override=current_override)
                if not main_input_include_context or (use_turn_level_flow and not main_input_include_context_when_turn_level)
                else build_model_text(
                    {**sample, "utterance": current_override}
                    if current_override is not None else sample,
                    include_context=True,
                )
                for sample in batch
            ]
            speaker_texts = [
                build_speaker_state_text(
                    {**sample, "utterance": current_override}
                    if current_override is not None else sample,
                    include_speaker_history=True,
                )
                for sample in batch
            ]
            previous_texts = [sample.get("previous_turn_text") or placeholder for sample in batch]

            enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            spk = tokenizer(speaker_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            prev_enc = tokenizer(previous_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc_ids, enc_mask = enc["input_ids"], enc["attention_mask"]
            spk_ids, spk_mask = spk["input_ids"], spk["attention_mask"]
            prev_ids, prev_mask = prev_enc["input_ids"], prev_enc["attention_mask"]

        labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
        if use_gold_previous_emotion:
            prev = torch.tensor([sample.get("previous_emotion_id", -100) for sample in batch], dtype=torch.long)
        else:
            prev = torch.full((len(batch),), -100, dtype=torch.long)
        previous_turn_exists = torch.tensor(
            [1 if sample.get("previous_turn_text") else 0 for sample in batch],
            dtype=torch.float32,
        )

        result = {
            "input_ids": enc_ids,
            "attention_mask": enc_mask,
            "speaker_input_ids": spk_ids,
            "speaker_attention_mask": spk_mask,
            "previous_input_ids": prev_ids,
            "previous_attention_mask": prev_mask,
            "previous_turn_exists": previous_turn_exists,
            "previous_emotion_ids": prev,
            "labels": labels,
            "meta": batch,
        }

        if use_turn_level_flow:
            turns_by_sample = [
                sample["_token_cache"]["context_input_ids"]
                if use_token_cache else _turn_texts(sample, include_context=True, current_utterance_override=current_override)
                for sample in batch
            ]
            max_turns = max(len(turns) for turns in turns_by_sample)
            flat_turns, turn_mask, current_turn_indices = [], [], []
            turn_speaker_ids = []
            for turns in turns_by_sample:
                num_turns = len(turns)
                current_turn_indices.append(num_turns - 1)
                if use_token_cache:
                    flat_turns.extend(turns + [[pad_token_id]] * (max_turns - num_turns))
                else:
                    flat_turns.extend(turns + [placeholder] * (max_turns - num_turns))
                turn_mask.append([1] * num_turns + [0] * (max_turns - num_turns))
                speaker_ids = _speaker_ids_for_turns(batch[len(turn_speaker_ids)], include_context=True)
                speaker_ids = speaker_ids + [0] * (max_turns - len(speaker_ids))
                turn_speaker_ids.append(speaker_ids)

            if use_token_cache:
                turn_input_ids, turn_attention_mask = _pad_sequences(flat_turns, pad_token_id)
            else:
                turn_enc = tokenizer(
                    flat_turns,
                    padding=True,
                    truncation=True,
                    max_length=turn_max_length,
                    return_tensors="pt",
                )
                turn_input_ids = turn_enc["input_ids"]
                turn_attention_mask = turn_enc["attention_mask"]
            batch_size = len(batch)
            result.update({
                "context_input_ids": turn_input_ids.view(batch_size, max_turns, -1),
                "context_attention_mask": turn_attention_mask.view(batch_size, max_turns, -1),
                "context_turn_mask": torch.tensor(turn_mask, dtype=torch.float32),
                "current_turn_indices": torch.tensor(current_turn_indices, dtype=torch.long),
                "context_turn_speaker_ids": torch.tensor(turn_speaker_ids, dtype=torch.long),
            })

        return result

    return collate
