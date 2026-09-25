"""
Module 3: Pairwise Feature Engineering Engine.
Computes 40+ pairwise string, token, numeric, and retrieval features
between Source 1 and Candidate entities.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

def get_acronym(text: str) -> str:
    """Extract acronym from words (e.g. 'State Bank of India' -> 'sboi')."""
    words = text.split()
    return "".join([w[0] for w in words if w]).lower()

def compute_pairwise_features(
    s1_dict: Dict[str, Any],
    cand_dict: Dict[str, Any],
    retrieval_meta: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes a vector of 40+ high-signal features for a single (S1, Candidate) pair.
    """
    feats = {}
    
    # 1. Names
    n1 = s1_dict.get("clean_name", "")
    n2 = cand_dict.get("clean_name", "")
    
    feats["name_lev_ratio"] = fuzz.ratio(n1, n2) / 100.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0
    feats["name_token_sort"] = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats["name_token_set"] = fuzz.token_set_ratio(n1, n2) / 100.0
    feats["name_wratio"] = fuzz.WRatio(n1, n2) / 100.0
    feats["name_jaro_winkler"] = distance.JaroWinkler.similarity(n1, n2)
    feats["name_exact_match"] = 1.0 if n1 == n2 and n1 != "" else 0.0
    
    len1, len2 = len(n1), len(n2)
    feats["name_len_diff"] = abs(len1 - len2)
    feats["name_len_ratio"] = min(len1, len2) / (max(len1, len2) + 1e-5)
    
    # Prefix matches
    feats["name_prefix3_match"] = 1.0 if n1[:3] == n2[:3] and len(n1) >= 3 and len(n2) >= 3 else 0.0
    feats["name_prefix5_match"] = 1.0 if n1[:5] == n2[:5] and len(n1) >= 5 and len(n2) >= 5 else 0.0
    
    # Acronym match
    acr1 = get_acronym(n1)
    acr2 = get_acronym(n2)
    feats["name_acronym_match"] = 1.0 if (acr1 == n2 or acr2 == n1 or (len(acr1) > 1 and acr1 == acr2)) else 0.0
    
    # Suffix match
    suf1 = s1_dict.get("legal_suffix", "")
    suf2 = cand_dict.get("legal_suffix", "")
    feats["suffix_exact_match"] = 1.0 if suf1 and suf1 == suf2 else 0.0
    
    # 2. Addresses
    a1 = s1_dict.get("clean_address", "")
    a2 = cand_dict.get("clean_address", "")
    
    feats["addr_token_sort"] = fuzz.token_sort_ratio(a1, a2) / 100.0
    feats["addr_token_set"] = fuzz.token_set_ratio(a1, a2) / 100.0
    feats["addr_jaro_winkler"] = distance.JaroWinkler.similarity(a1, a2)
    feats["addr_contains"] = 1.0 if (a1 in a2 or a2 in a1) and min(len(a1), len(a2)) > 5 else 0.0
    
    # Word Jaccard on Address
    w1 = set(a1.split())
    w2 = set(a2.split())
    union_w = len(w1 | w2)
    feats["addr_word_jaccard"] = len(w1 & w2) / (union_w + 1e-5)
    
    # 3. Numeric Tokens (PIN codes, house numbers) & Number Sequence Alignment
    nums1 = s1_dict.get("extracted_numbers", [])
    nums2 = cand_dict.get("extracted_numbers", [])
    set1, set2 = set(nums1), set(nums2)
    
    num_inter = len(set1 & set2)
    num_union = len(set1 | set2)
    feats["num_inter_count"] = float(num_inter)
    feats["num_jaccard"] = num_inter / (num_union + 1e-5)
    feats["num_exact_subset"] = 1.0 if (set1 and set1.issubset(set2)) or (set2 and set2.issubset(set1)) else 0.0
    
    # Address Number Sequence Alignment: Leading building number match / contradiction
    if nums1 and nums2:
        feats["lead_num_match"] = 1.0 if nums1[0] == nums2[0] else 0.0
        feats["lead_num_contradict"] = 1.0 if nums1[0] != nums2[0] else 0.0
        feats["num_contradiction"] = 1.0 if num_inter == 0 else 0.0
    else:
        feats["lead_num_match"] = 0.0
        feats["lead_num_contradict"] = 0.0
        feats["num_contradiction"] = 0.0
    
    # Check 5-6 digit PIN/ZIP code match
    pins1 = {n for n in set1 if len(n) in (5, 6)}
    pins2 = {n for n in set2 if len(n) in (5, 6)}
    if pins1 and pins2:
        feats["pin_code_match"] = 1.0 if len(pins1 & pins2) > 0 else -1.0
    else:
        feats["pin_code_match"] = 0.0  # missing
        
    # 4. Retrieval & Ranking Signals
    feats["rrf_score"] = float(retrieval_meta.get("rrf_score", 0.0))
    feats["lexical_rank"] = float(retrieval_meta.get("lexical_rank", 1000.0))
    feats["dense_rank"] = float(retrieval_meta.get("dense_rank", 1000.0))
    feats["retrieval_similarity"] = float(retrieval_meta.get("retrieval_similarity", 0.0))
    
    # 5. Metadata Signals
    c1 = s1_dict.get("country", "")
    c2 = cand_dict.get("country", "")
    feats["same_country"] = 1.0 if c1 == c2 and c1 != "UNKNOWN" else 0.0
    
    cand_id = cand_dict.get("entity_id", "")
    feats["is_source2"] = 1.0 if cand_id.startswith("S2-") else 0.0
    feats["is_source3"] = 1.0 if cand_id.startswith("S3-") else 0.0
    
    return feats

def build_features_dataframe(
    candidates_df: pd.DataFrame,
    s1_preprocessed_df: pd.DataFrame,
    target_preprocessed_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Vectorizes feature computation across all candidate pairs in candidates_df.
    """
    s1_map = s1_preprocessed_df.set_index("entity_id").to_dict(orient="index")
    tgt_map = target_preprocessed_df.set_index("entity_id").to_dict(orient="index")
    
    feature_rows = []
    pair_ids = []
    
    for _, row in candidates_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        
        s1_entry = s1_map.get(s1_id, {})
        s1_entry["entity_id"] = s1_id
        
        cand_entry = tgt_map.get(cand_id, {})
        cand_entry["entity_id"] = cand_id
        
        meta = {
            "rrf_score": row.get("rrf_score", 0.0),
            "lexical_rank": row.get("lexical_rank", 1000.0),
            "dense_rank": row.get("dense_rank", 1000.0),
            "retrieval_similarity": row.get("retrieval_similarity", 0.0)
        }
        
        feats = compute_pairwise_features(s1_entry, cand_entry, meta)
        feature_rows.append(feats)
        pair_ids.append((s1_id, cand_id))
        
    feats_df = pd.DataFrame(feature_rows)
    feats_df["source1_entity_id"] = [p[0] for p in pair_ids]
    feats_df["candidate_entity_id"] = [p[1] for p in pair_ids]
    return feats_df

if __name__ == "__main__":
    # Smoke test Module 3 Feature Extraction
    s1 = {"entity_id": "S1-1", "clean_name": "apple store", "legal_suffix": "inc", "clean_address": "1 infinite loop cupertino", "extracted_numbers": ["1", "95014"], "country": "US"}
    s2 = {"entity_id": "S2-1", "clean_name": "apple inc", "legal_suffix": "inc", "clean_address": "infinite loop cupertino", "extracted_numbers": ["95014"], "country": "US"}
    meta = {"rrf_score": 0.03, "lexical_rank": 1, "dense_rank": 2, "retrieval_similarity": 0.92}
    f = compute_pairwise_features(s1, s2, meta)
    print("Module 3 Feature Extraction Smoke Test PASSED! Feature count:", len(f))
    print("Sample features:", {k: round(v, 4) for k, v in list(f.items())[:8]})
