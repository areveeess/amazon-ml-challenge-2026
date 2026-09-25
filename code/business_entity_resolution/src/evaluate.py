"""
Module 4 (Part 1): Official Macro F0.5 Evaluation Metric & Threshold Optimizer.
Directly implements the competition evaluation criteria:
- F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Macro-averaged across all Source 1 entities
- Strict singleton handling:
    * True empty + Predicted empty -> 1.0
    * True empty + Predicted non-empty -> 0.0
    * True non-empty + Predicted empty -> 0.0
- Grid/Coordinate search to find optimal (tau_singleton, tau_match)
"""

from typing import Dict, List, Set, Tuple
import numpy as np

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity."""
    # Singleton check
    if not true_matches:
        return 1.0 if not pred_matches else 0.0
    
    if not pred_matches:
        return 0.0
        
    true_pos = len(true_matches & pred_matches)
    if true_pos == 0:
        return 0.0
        
    precision = true_pos / len(pred_matches)
    recall = true_pos / len(true_matches)
    
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
        
    return (1.25 * precision * recall) / denom

def compute_macro_f05(
    ground_truth_map: Dict[str, Set[str]],
    predictions_map: Dict[str, Set[str]]
) -> float:
    """
    Macro-averages F_0.5 across all Source 1 entities in ground truth.
    ground_truth_map: {s1_id: set([matched_ids])}
    predictions_map: {s1_id: set([matched_ids])}
    """
    scores = []
    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        f05 = compute_entity_f05(true_set, pred_set)
        scores.append(f05)
        
    return float(np.mean(scores)) if scores else 0.0

def optimize_gating_thresholds(
    scored_pairs: List[Tuple[str, str, float]],
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Tuple[float, float, float]:
    """
    Searches for optimal (tau_singleton, tau_match) maximizing macro F_0.5.
    Returns (best_f05, best_tau_singleton, best_tau_match)
    """
    from collections import defaultdict
    s1_candidates = defaultdict(list)
    for s1_id, cand_id, prob in scored_pairs:
        s1_candidates[s1_id].append((cand_id, prob))
        
    best_f05 = -1.0
    best_tau_singleton = 0.65
    best_tau_match = 0.70
    
    # Grid search over conservative threshold ranges
    for tau_sing in np.linspace(0.45, 0.85, 9):
        for tau_m in np.linspace(0.50, 0.90, 9):
            if tau_m < tau_sing - 0.1:
                continue
                
            pred_map = {}
            for s1_id in all_s1_ids:
                cands = s1_candidates.get(s1_id, [])
                if not cands:
                    pred_map[s1_id] = set()
                    continue
                    
                max_p = max([p for _, p in cands])
                # Singleton Gate
                if max_p < tau_sing:
                    pred_map[s1_id] = set()
                else:
                    # Match Gate
                    matches = {cid for cid, p in cands if p >= tau_m}
                    pred_map[s1_id] = matches
                    
            f05 = compute_macro_f05(ground_truth_map, pred_map)
            if f05 > best_f05:
                best_f05 = f05
                best_tau_singleton = tau_sing
                best_tau_match = tau_m
                
    print(f"[Module 4 Optimizer] Optimal Macro F0.5 = {best_f05:.4f} (tau_singleton = {best_tau_singleton:.2f}, tau_match = {best_tau_match:.2f})")
    return best_f05, best_tau_singleton, best_tau_match

if __name__ == "__main__":
    # Smoke test metric against problem statement example:
    # S1-00001: True = [S2-00047, S3-00812], Pred = [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 1.0, Expected F_0.5 = 0.714
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    f05 = compute_entity_f05(true_set, pred_set)
    print(f"Problem Statement Example F_0.5: {f05:.3f} (Expected: 0.714)")
    assert abs(f05 - 0.714) < 0.005, "F0.5 formula discrepancy!"
    
    # Singleton test
    assert compute_entity_f05(set(), set()) == 1.0, "Singleton empty should score 1.0"
    assert compute_entity_f05(set(), {"S2-999"}) == 0.0, "Singleton with false merge should score 0.0"
    print("Module 4 Evaluation Smoke Test PASSED!")
