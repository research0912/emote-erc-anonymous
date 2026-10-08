import json
from typing import Dict, Any, List

def generate_natural_language_explanation(xai_results: Dict[str, Any]) -> str:
    """
    Generates a beautifully structured human-readable explanation from the consolidated XAI results
    adhering to the requested template in xai_new_things.txt.
    """
    pred_emotion = str(xai_results.get("predicted_emotion", "UNKNOWN")).upper()
    confidence = xai_results.get("confidence", 0.0)
    
    # 1. Format Primary Evidence
    contribs = xai_results.get("stream_contributions", {})
    evidence_lines = []
    # Map friendly names for presentation
    name_mapping = {
        "text": "Utterance text",
        "dialogue_flow": "Dialogue flow history",
        "gnn": "Graph dialogue network",
        "speaker_state": "Speaker state history",
        "emotion_shift": "Emotion shift tracking"
    }
    # Sort contributions by value descending
    sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
    for k, v in sorted_contribs:
        label = name_mapping.get(k, k.replace("_", " ").capitalize())
        evidence_lines.append(f"- {label} contributed {int(v * 100)}%")
        
    primary_evidence_text = "\n".join(evidence_lines)
    
    # 2. Format Most Influential Context
    turning_points = xai_results.get("turning_points", [])
    if turning_points:
        top_tp = turning_points[0]
        # Check if importance is greater than 0
        if top_tp.get("importance", 0.0) > 0:
            influential_context = f"Turn {top_tp.get('turn_id', 0)} (\"{top_tp.get('context_turn', '')}\")"
        else:
            influential_context = "None (context turns did not heavily impact prediction probability)"
    else:
        influential_context = "None"
        
    # 3. Format Most Influential Words (Lexical occlusion)
    token_occlusions = xai_results.get("token_occlusions", [])
    # Sort token drops by pred_probability_drop descending
    sorted_tokens = sorted(token_occlusions, key=lambda x: x.get("pred_probability_drop", x.get("gold_probability_drop", 0.0)), reverse=True)
    influential_words = []
    for tok in sorted_tokens[:3]:
        influential_words.append(f"\"{tok.get('token')}\"")
    if not influential_words:
        influential_words = ["None"]
        
    # 4. Format Detected Shift
    drift_type = xai_results.get("emotion_drift", {}).get("shift_type", "Stable")
    
    # 5. Format the Text Template
    explanation_text = f"""The model predicted {pred_emotion}.

Primary evidence:
{primary_evidence_text}

Most influential context:
{influential_context}

Most influential words:
{", ".join(influential_words)}

Detected:
{drift_type} emotional shift

Confidence:
{confidence:.2f}
"""
    return explanation_text
