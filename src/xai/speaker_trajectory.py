import torch
import torch.nn.functional as F
import pandas as pd
from typing import List, Dict, Any

def analyze_speaker_trajectory(model, tokenizer, samples: List[Dict[str, Any]], labels: List[str], device: str, max_length: int) -> pd.DataFrame:
    """
    Groups samples by dialogue and speaker, extracts chronological speaker embeddings,
    computes cosine similarity between consecutive speaker state embeddings of the same speaker,
    and returns a DataFrame representing the speaker's emotional trajectory.
    """
    from src.xai.faithfulness import _predict_one
    from collections import defaultdict
    
    # 1. Group samples by dialogue
    dialogue_samples = defaultdict(list)
    for s in samples:
        dialogue_samples[s.get("dialogue_id")].append(s)
        
    rows = []
    
    # 2. Process each dialogue
    for dialogue_id, turns in dialogue_samples.items():
        # Sort turns by turn_id
        turns = sorted(turns, key=lambda x: x.get("turn_id", 0))
        
        # Track last embedding and emotion per speaker
        speaker_last_embedding = {}
        speaker_last_emotion = {}
        
        for turn in turns:
            speaker = turn.get("speaker")
            gold_emotion = turn.get("label_name")
            turn_id = turn.get("turn_id")
            
            # Predict and capture the embeddings
            # To get speaker embeddings, we run a prediction on this sample
            use_turn_level_flow = model.cfg.get("model", {}).get("dialogue_flow_source") == "turn_level"
            from src.xai.faithfulness import _current_text, build_speaker_state_text, _turn_inputs
            from src.data_loader.base import build_model_text
            
            text = _current_text(turn) if use_turn_level_flow else build_model_text(turn, include_context=True)
            spk_text = build_speaker_state_text(turn, include_speaker_history=True)
            has_previous_turn = bool(turn.get("previous_turn_text"))
            previous_text = turn.get("previous_turn_text") or tokenizer.unk_token or "[NO_PREVIOUS_TURN]"
            
            enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            spk = tokenizer([spk_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            prev_enc = tokenizer([previous_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            prev = torch.tensor([turn.get("previous_emotion_id", -100)], dtype=torch.long, device=device)
            previous_turn_exists = torch.tensor([1.0 if has_previous_turn else 0.0], dtype=torch.float32, device=device)
            
            turn_inputs = _turn_inputs(model, tokenizer, turn, device, include_context=True) if use_turn_level_flow else {}
            
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    speaker_input_ids=spk["input_ids"], speaker_attention_mask=spk["attention_mask"],
                    previous_input_ids=prev_enc["input_ids"], previous_attention_mask=prev_enc["attention_mask"],
                    previous_turn_exists=previous_turn_exists,
                    previous_emotion_ids=prev,
                    **turn_inputs,
                )
                
            # Extract speaker state representation
            speaker_state = out["evidence"].get("speaker_state_repr")
            
            # Calculate cosine similarity and emotion drift
            trajectory_score = 1.0
            emotion_drift = f"START -> {gold_emotion}"
            
            if speaker_state is not None:
                current_emb = speaker_state[0].float() # [dim]
                if speaker in speaker_last_embedding:
                    prev_emb = speaker_last_embedding[speaker]
                    # Cosine similarity
                    sim = F.cosine_similarity(current_emb.unsqueeze(0), prev_emb.unsqueeze(0))
                    trajectory_score = float(sim.item())
                    
                    prev_emo = speaker_last_emotion[speaker]
                    emotion_drift = f"{prev_emo} -> {gold_emotion}"
                
                # Update history
                speaker_last_embedding[speaker] = current_emb
                speaker_last_emotion[speaker] = gold_emotion
            
            rows.append({
                "dialogue_id": dialogue_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "gold_emotion": gold_emotion,
                "trajectory_score": trajectory_score,
                "emotion_drift": emotion_drift
            })
            
    return pd.DataFrame(rows)
