import re
import numpy as np
from copy import deepcopy
from typing import Dict, Any, List
from src.xai.faithfulness import _predict_one

def _spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Computes Spearman's rank correlation coefficient between two lists."""
    n = len(x)
    if n <= 1:
        return 1.0
        
    # Get ranks
    x_ranks = np.argsort(np.argsort(x))
    y_ranks = np.argsort(np.argsort(y))
    
    d_squared = np.sum((x_ranks - y_ranks) ** 2)
    rs = 1.0 - (6.0 * d_squared) / (n * (n**2 - 1))
    return float(rs)

def analyze_explanation_consistency(model, tokenizer, sample: Dict[str, Any], labels: List[str], device: str, max_length: int, max_words: int = 15, num_lime_samples: int = 25, num_shap_samples: int = 15) -> Dict[str, Any]:
    """
    Computes custom local SHAP, LIME, and Occlusion importances for the tokens in the utterance,
    ranks them, and calculates Spearman's Rank Correlation between the explainers.
    """
    base = _predict_one(model, tokenizer, sample, labels, device, max_length)
    original_pred = base["pred"]
    gold_id = labels.index(original_pred)
    base_prob = base["gold_probability"]
    
    mask_token = tokenizer.mask_token or tokenizer.unk_token or "[MASK]"
    
    # Split utterance into words
    words = re.findall(r"\w+|[^\w\s]", sample["utterance"])
    words = words[:max_words]
    n = len(words)
    
    if n == 0:
        return {"lime_shap": 1.0, "lime_occlusion": 1.0, "shap_occlusion": 1.0}
        
    # Helper to evaluate a binary mask of active words
    def evaluate_mask(mask: np.ndarray) -> float:
        masked_words = [words[i] if mask[i] == 1 else mask_token for i in range(n)]
        modified = deepcopy(sample)
        modified["utterance"] = " ".join(masked_words)
        pred = _predict_one(model, tokenizer, modified, labels, device, max_length)
        return pred["probabilities"][gold_id]
        
    # 1. OCCLUSION IMPORTANCES
    occlusion_scores = []
    for i in range(n):
        mask = np.ones(n)
        mask[i] = 0
        prob = evaluate_mask(mask)
        occlusion_scores.append(base_prob - prob)
        
    # 2. LIME IMPORTANCES
    # Generate random binary masks (perturbations)
    # Using passed num_lime_samples parameter
    lime_masks = np.random.binomial(1, 0.7, size=(num_lime_samples, n))
    # Make sure base mask and empty mask are not included or handled
    lime_targets = []
    for mask in lime_masks:
        prob = evaluate_mask(mask)
        lime_targets.append(prob - base_prob)
        
    # Fit simple linear ridge regression to compute feature coefficients
    X = lime_masks
    y = np.array(lime_targets)
    # Ridge formula: w = (X^T X + alpha*I)^(-1) X^T y
    alpha = 1.0
    XTX = X.T @ X
    I = np.eye(n)
    try:
        w = np.linalg.solve(XTX + alpha * I, X.T @ y)
        lime_scores = w.tolist()
    except Exception:
        # Fallback to simple correlation
        lime_scores = [float(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 0 else 0.0 for i in range(n)]
        
    # 3. SHAP IMPORTANCES (Sampling-based Shapley values)
    # Using passed num_shap_samples parameter
    shap_scores = [0.0] * n
    for i in range(n):
        marginal_contribs = []
        for _ in range(num_shap_samples):
            # Sample a subset S of other features
            subset = np.random.binomial(1, 0.5, size=n)
            subset[i] = 0 # Ensure target feature is not in S
            
            # S with target active
            mask_with = subset.copy()
            mask_with[i] = 1
            
            # S without target
            mask_without = subset.copy()
            mask_without[i] = 0
            
            val_with = evaluate_mask(mask_with)
            val_without = evaluate_mask(mask_without)
            marginal_contribs.append(val_with - val_without)
        shap_scores[i] = float(np.mean(marginal_contribs))
        
    # Calculate Spearman rank correlations (using absolute value for [0, 1] scale)
    lime_shap = abs(_spearman_rank_correlation(lime_scores, shap_scores))
    lime_occlusion = abs(_spearman_rank_correlation(lime_scores, occlusion_scores))
    shap_occlusion = abs(_spearman_rank_correlation(shap_scores, occlusion_scores))
    
    # Handle NaN or constant scores
    if np.isnan(lime_shap): lime_shap = 0.0
    if np.isnan(lime_occlusion): lime_occlusion = 0.0
    if np.isnan(shap_occlusion): shap_occlusion = 0.0
    
    return {
        "lime_shap": round(lime_shap, 4),
        "lime_occlusion": round(lime_occlusion, 4),
        "shap_occlusion": round(shap_occlusion, 4)
    }
