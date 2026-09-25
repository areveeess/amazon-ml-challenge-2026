"""
Module 4 (Part 3): Calibrated Two-Stage Gating & Submission TSV Formatter.
Implements:
- Singleton Gating (protects 1.0 score for true singletons)
- Match Selection Thresholding
- Generation of output/matching_results.tsv satisfying all competition rules
"""

import os
from collections import defaultdict
from typing import Dict, List, Set
import pandas as pd

def apply_two_stage_gating(
    scored_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    tau_singleton: float = 0.65,
    tau_match: float = 0.70
) -> Dict[str, List[str]]:
    """
    Applies Two-Stage Gating across all Source 1 entities:
    1. If an entity's top candidate probability < tau_singleton:
       Declare as Singleton (return empty list []).
    2. Otherwise:
       Select all candidates with probability >= tau_match.
    """
    s1_candidates = defaultdict(list)
    for _, row in scored_pairs_df.iterrows():
        s1 = str(row["source1_entity_id"])
        cand = str(row["candidate_entity_id"])
        prob = float(row["predicted_prob"])
        s1_candidates[s1].append((cand, prob))
        
    predictions = {}
    
    for s1_id in all_s1_ids:
        cands = s1_candidates.get(s1_id, [])
        if not cands:
            predictions[s1_id] = []
            continue
            
        max_p = max([p for _, p in cands])
        
        # Stage 1: Singleton Gate
        if max_p < tau_singleton:
            predictions[s1_id] = []
        else:
            # Stage 2: Match Gate
            # Sort matches descending by probability
            matched = [cid for cid, p in sorted(cands, key=lambda x: x[1], reverse=True) if p >= tau_match]
            # Deduplicate preserving order
            seen = set()
            dedup_matched = []
            for m in matched:
                if m not in seen:
                    seen.add(m)
                    dedup_matched.append(m)
            predictions[s1_id] = dedup_matched
            
    return predictions

def export_matching_results_tsv(
    predictions_map: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_path: str
):
    """
    Writes matching_results.tsv in strict competition format:
    source1_entity_id\tmatched_entity_ids
    (single tab separator, comma-separated ID list, empty for singletons, zero quoting)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = predictions_map.get(s1_id, [])
            match_str = ",".join(matches)
            f.write(f"{s1_id}\t{match_str}\n")
            
    print(f"[Module 4] Successfully generated leaderboard file: {output_path}")

if __name__ == "__main__":
    preds = {"S1-001": ["S2-047", "S3-812"], "S1-002": []}
    export_matching_results_tsv(preds, ["S1-001", "S1-002"], "test_matching.tsv")
    with open("test_matching.tsv") as f:
        print(f.read())
    os.remove("test_matching.tsv")
    print("Module 4 Postprocessing Smoke Test PASSED!")
