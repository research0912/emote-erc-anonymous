import torch
import torch.nn.functional as F
from typing import Dict, List, Any

def compute_emotion_centroids(model, tokenizer, ref_samples: List[Dict[str, Any]], labels: List[str], device: str, max_length: int) -> Dict[str, torch.Tensor]:
    """
    Computes class centroids (mean gated feature representations) for each emotion
    using a set of reference samples.
    """
    from src.xai.faithfulness import _current_text, build_speaker_state_text, _turn_inputs
    from src.data_loader.base import build_model_text
    
    # Accumulate features per emotion
    emotion_features = {label: [] for label in labels}
    
    # We only need a moderate number of reference samples to compute stable centroids
    max_ref = min(len(ref_samples), 100)
    for sample in ref_samples[:max_ref]:
        gold_emo = sample.get("label_name")
        if gold_emo not in emotion_features:
            continue
            
        use_turn_level_flow = model.cfg.get("model", {}).get("dialogue_flow_source") == "turn_level"
        text = _current_text(sample) if use_turn_level_flow else build_model_text(sample, include_context=True)
        spk_text = build_speaker_state_text(sample, include_speaker_history=True)
        has_previous_turn = bool(sample.get("previous_turn_text"))
        previous_text = sample.get("previous_turn_text") or tokenizer.unk_token or "[NO_PREVIOUS_TURN]"
        
        enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        spk = tokenizer([spk_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        prev_enc = tokenizer([previous_text], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        prev = torch.tensor([sample.get("previous_emotion_id", -100)], dtype=torch.long, device=device)
        previous_turn_exists = torch.tensor([1.0 if has_previous_turn else 0.0], dtype=torch.float32, device=device)
        
        turn_inputs = _turn_inputs(model, tokenizer, sample, device, include_context=True) if use_turn_level_flow else {}
        
        with torch.no_grad():
            out = model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                speaker_input_ids=spk["input_ids"], speaker_attention_mask=spk["attention_mask"],
                previous_input_ids=prev_enc["input_ids"], previous_attention_mask=prev_enc["attention_mask"],
                previous_turn_exists=previous_turn_exists,
                previous_emotion_ids=prev,
                **turn_inputs,
            )
            # Capture gated features
            gated = out["features"][0].cpu() # [fusion_dim]
            emotion_features[gold_emo].append(gated)
            
    # Compute mean centroid vector for each emotion
    centroids = {}
    # Use model's overall mean across all samples as a fallback if an emotion is empty
    all_vectors = [v for lst in emotion_features.values() for v in lst]
    if all_vectors:
        fallback_centroid = torch.stack(all_vectors).mean(dim=0)
    else:
        # Fallback to zeros if no vectors at all
        # Dynamic inspection of model classifier input size to find the dimension
        dim = 768
        for m in model.classifier.children():
            if isinstance(m, torch.nn.Linear):
                dim = m.in_features
                break
        fallback_centroid = torch.zeros(dim)
        
    for label in labels:
        vectors = emotion_features[label]
        if vectors:
            centroids[label] = torch.stack(vectors).mean(dim=0)
        else:
            centroids[label] = fallback_centroid.clone()
            
    return centroids

def match_emotion_prototype(centroids: Dict[str, torch.Tensor], sample_feature: torch.Tensor) -> Dict[str, Any]:
    """
    Computes distance (Euclidean and Cosine) from the sample feature vector to each class centroid,
    and returns the closest emotion along with distance breakdown.
    """
    if sample_feature.dim() > 1:
        sample_feature = sample_feature[0]
    sample_feat_cpu = sample_feature.cpu().float()
    
    distances = {}
    cosine_sims = {}
    
    for emo, centroid in centroids.items():
        centroid_cpu = centroid.float()
        # Euclidean distance
        dist = float(torch.norm(sample_feat_cpu - centroid_cpu, p=2).item())
        distances[emo] = round(dist, 4)
        
        # Cosine similarity
        cos_sim = float(F.cosine_similarity(sample_feat_cpu.unsqueeze(0), centroid_cpu.unsqueeze(0)).item())
        cosine_sims[emo] = round(cos_sim, 4)
        
    closest_emotion = min(distances, key=distances.get)
    return {
        "closest_emotion": closest_emotion,
        "distance": distances[closest_emotion],
        "distances": distances,
        "cosine_similarities": cosine_sims
    }
