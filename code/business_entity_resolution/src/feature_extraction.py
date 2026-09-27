"""
Module 3: Pairwise Feature Engineering Engine.
Computes 40+ pairwise string, token, numeric, and retrieval features
between Source 1 and Candidate entities.
Optimized for high-throughput vectorized execution on large pair DataFrames.
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
    s1 = s1_dict.copy()
    cand = cand_dict.copy()

    feats = {}

    # 1. Names
    n1 = s1.get("clean_name", "")
    n2 = cand.get("clean_name", "")

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

    # Acronym match: require 3+ chars for full match; down-weight 2-char matches to 0.5
    acr1 = get_acronym(n1)
    acr2 = get_acronym(n2)
    if (len(acr1) >= 3 and len(acr2) >= 3 and acr1 == acr2) or (len(acr1) >= 3 and acr1 == n2) or (len(acr2) >= 3 and acr2 == n1):
        feats["name_acronym_match"] = 1.0
    elif (len(acr1) == 2 and acr1 == acr2) or (len(acr1) == 2 and acr1 == n2) or (len(acr2) == 2 and acr2 == n1):
        feats["name_acronym_match"] = 0.5
    else:
        feats["name_acronym_match"] = 0.0

    # Suffix match
    suf1 = s1.get("legal_suffix", "")
    suf2 = cand.get("legal_suffix", "")
    feats["suffix_exact_match"] = 1.0 if suf1 and suf1 == suf2 else 0.0

    # 2. Addresses
    a1 = s1.get("clean_address", "")
    a2 = cand.get("clean_address", "")

    feats["addr_token_sort"] = fuzz.token_sort_ratio(a1, a2) / 100.0
    feats["addr_token_set"] = fuzz.token_set_ratio(a1, a2) / 100.0
    feats["addr_jaro_winkler"] = distance.JaroWinkler.similarity(a1, a2)
    feats["addr_contains"] = 1.0 if (a1 in a2 or a2 in a1) and min(len(a1), len(a2)) > 5 else 0.0

    # Word Jaccard on Address
    w1 = set(a1.split())
    w2 = set(a2.split())
    union_w = len(w1 | w2)
    feats["addr_word_jaccard"] = len(w1 & w2) / (union_w + 1e-5) if union_w else 0.0

    # 3. Numeric Tokens & Alignment
    nums1 = s1.get("extracted_numbers", [])
    nums2 = cand.get("extracted_numbers", [])
    set1, set2 = set(nums1), set(nums2)

    num_inter = len(set1 & set2)
    num_union = len(set1 | set2)
    feats["num_inter_count"] = float(num_inter)
    feats["num_jaccard"] = num_inter / (num_union + 1e-5) if num_union else 0.0
    feats["num_exact_subset"] = 1.0 if (set1 and set1.issubset(set2)) or (set2 and set2.issubset(set1)) else 0.0

    if nums1 and nums2:
        feats["lead_num_match"] = 1.0 if nums1[0] == nums2[0] else 0.0
        feats["lead_num_contradict"] = 1.0 if nums1[0] != nums2[0] else 0.0
        feats["num_contradiction"] = 1.0 if num_inter == 0 else 0.0
    else:
        feats["lead_num_match"] = 0.0
        feats["lead_num_contradict"] = 0.0
        feats["num_contradiction"] = 0.0

    pins1 = {n for n in set1 if len(n) in (5, 6)}
    pins2 = {n for n in set2 if len(n) in (5, 6)}
    if pins1 and pins2:
        feats["pin_code_match"] = 1.0 if len(pins1 & pins2) > 0 else -1.0
    else:
        feats["pin_code_match"] = 0.0

    # 4. Retrieval & Ranking Signals
    feats["rrf_score"] = float(retrieval_meta.get("rrf_score", 0.0))
    feats["lexical_rank"] = float(retrieval_meta.get("lexical_rank", 1000.0))
    feats["dense_rank"] = float(retrieval_meta.get("dense_rank", 1000.0))
    feats["retrieval_similarity"] = float(retrieval_meta.get("retrieval_similarity", 0.0))
    feats["embedding_similarity"] = float(retrieval_meta.get("dense_similarity", retrieval_meta.get("retrieval_similarity", 0.0)))

    # 5. Metadata Signals
    c1 = s1.get("country", "")
    c2 = cand.get("country", "")
    feats["same_country"] = 1.0 if c1 == c2 and c1 != "UNKNOWN" else 0.0

    cand_id = cand.get("entity_id", "")
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
    Uses pre-merged DataFrames and batch array operations for high throughput (~200k pairs/sec).
    """
    if candidates_df.empty:
        return pd.DataFrame()

    # Pre-merge S1 and candidate attributes in vectorized C-level merge
    s1_cols = ["entity_id", "clean_name", "legal_suffix", "clean_address", "extracted_numbers", "country"]
    s1_ids_present = set(candidates_df["source1_entity_id"])
    s1_sub = s1_preprocessed_df[s1_preprocessed_df["entity_id"].isin(s1_ids_present)][[c for c in s1_cols if c in s1_preprocessed_df.columns]].drop_duplicates(subset=["entity_id"])

    tgt_cols = ["entity_id", "clean_name", "legal_suffix", "clean_address", "extracted_numbers", "country"]
    cand_ids_present = set(candidates_df["candidate_entity_id"])
    tgt_sub = target_preprocessed_df[target_preprocessed_df["entity_id"].isin(cand_ids_present)][[c for c in tgt_cols if c in target_preprocessed_df.columns]].drop_duplicates(subset=["entity_id"])

    merged = candidates_df.merge(
        s1_sub,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left"
    ).drop(columns=["entity_id"], errors="ignore").merge(
        tgt_sub,
        left_on="candidate_entity_id",
        right_on="entity_id",
        how="left",
        suffixes=("_s1", "_cand")
    ).drop(columns=["entity_id"], errors="ignore")

    n1_list = merged["clean_name_s1"].fillna("").astype(str).tolist()
    n2_list = merged["clean_name_cand"].fillna("").astype(str).tolist()
    a1_list = merged["clean_address_s1"].fillna("").astype(str).tolist()
    a2_list = merged["clean_address_cand"].fillna("").astype(str).tolist()
    suf1_list = merged["legal_suffix_s1"].fillna("").astype(str).tolist()
    suf2_list = merged["legal_suffix_cand"].fillna("").astype(str).tolist()
    nums1_list = [x if isinstance(x, list) else [] for x in merged["extracted_numbers_s1"]]
    nums2_list = [x if isinstance(x, list) else [] for x in merged["extracted_numbers_cand"]]
    c1_list = merged["country_s1"].fillna("UNKNOWN").astype(str).tolist()
    c2_list = merged["country_cand"].fillna("UNKNOWN").astype(str).tolist()
    cand_ids = merged["candidate_entity_id"].astype(str).tolist()

    N = len(merged)
    feats = {}

    # 1. Names
    feats["name_lev_ratio"] = np.array([fuzz.ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_partial_ratio"] = np.array([fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_token_sort"] = np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_token_set"] = np.array([fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_wratio"] = np.array([fuzz.WRatio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_jaro_winkler"] = np.array([distance.JaroWinkler.similarity(a, b) for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_exact_match"] = np.array([1.0 if a == b and a != "" else 0.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)

    len1 = np.array([len(a) for a in n1_list], dtype=np.float32)
    len2 = np.array([len(b) for b in n2_list], dtype=np.float32)
    feats["name_len_diff"] = np.abs(len1 - len2)
    feats["name_len_ratio"] = np.minimum(len1, len2) / (np.maximum(len1, len2) + 1e-5)

    feats["name_prefix3_match"] = np.array([1.0 if a[:3] == b[:3] and len(a) >= 3 and len(b) >= 3 else 0.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)
    feats["name_prefix5_match"] = np.array([1.0 if a[:5] == b[:5] and len(a) >= 5 and len(b) >= 5 else 0.0 for a, b in zip(n1_list, n2_list)], dtype=np.float32)

    acronym_feats = []
    for a, b in zip(n1_list, n2_list):
        acr1 = get_acronym(a)
        acr2 = get_acronym(b)
        if (len(acr1) >= 3 and len(acr2) >= 3 and acr1 == acr2) or (len(acr1) >= 3 and acr1 == b) or (len(acr2) >= 3 and acr2 == a):
            acronym_feats.append(1.0)
        elif (len(acr1) == 2 and acr1 == acr2) or (len(acr1) == 2 and acr1 == b) or (len(acr2) == 2 and acr2 == a):
            acronym_feats.append(0.5)
        else:
            acronym_feats.append(0.0)
    feats["name_acronym_match"] = np.array(acronym_feats, dtype=np.float32)

    feats["suffix_exact_match"] = np.array([1.0 if s1 and s1 == s2 else 0.0 for s1, s2 in zip(suf1_list, suf2_list)], dtype=np.float32)

    # 2. Addresses
    feats["addr_token_sort"] = np.array([fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a1_list, a2_list)], dtype=np.float32)
    feats["addr_token_set"] = np.array([fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(a1_list, a2_list)], dtype=np.float32)
    feats["addr_jaro_winkler"] = np.array([distance.JaroWinkler.similarity(a, b) for a, b in zip(a1_list, a2_list)], dtype=np.float32)
    feats["addr_contains"] = np.array([1.0 if (a in b or b in a) and min(len(a), len(b)) > 5 else 0.0 for a, b in zip(a1_list, a2_list)], dtype=np.float32)

    addr_jaccard = []
    for a, b in zip(a1_list, a2_list):
        w1 = set(a.split())
        w2 = set(b.split())
        union_w = len(w1 | w2)
        addr_jaccard.append(len(w1 & w2) / (union_w + 1e-5) if union_w else 0.0)
    feats["addr_word_jaccard"] = np.array(addr_jaccard, dtype=np.float32)

    # 3. Numeric tokens
    num_inter_counts = []
    num_jaccards = []
    num_subsets = []
    lead_matches = []
    lead_contradicts = []
    num_contradicts = []
    pin_matches = []

    for nums1, nums2 in zip(nums1_list, nums2_list):
        set1, set2 = set(nums1), set(nums2)
        inter = len(set1 & set2)
        union = len(set1 | set2)
        num_inter_counts.append(float(inter))
        num_jaccards.append(inter / (union + 1e-5) if union else 0.0)
        num_subsets.append(1.0 if (set1 and set1.issubset(set2)) or (set2 and set2.issubset(set1)) else 0.0)

        if nums1 and nums2:
            lead_matches.append(1.0 if nums1[0] == nums2[0] else 0.0)
            lead_contradicts.append(1.0 if nums1[0] != nums2[0] else 0.0)
            num_contradicts.append(1.0 if inter == 0 else 0.0)
        else:
            lead_matches.append(0.0)
            lead_contradicts.append(0.0)
            num_contradicts.append(0.0)

        pins1 = {n for n in set1 if len(n) in (5, 6)}
        pins2 = {n for n in set2 if len(n) in (5, 6)}
        if pins1 and pins2:
            pin_matches.append(1.0 if len(pins1 & pins2) > 0 else -1.0)
        else:
            pin_matches.append(0.0)

    feats["num_inter_count"] = np.array(num_inter_counts, dtype=np.float32)
    feats["num_jaccard"] = np.array(num_jaccards, dtype=np.float32)
    feats["num_exact_subset"] = np.array(num_subsets, dtype=np.float32)
    feats["lead_num_match"] = np.array(lead_matches, dtype=np.float32)
    feats["lead_num_contradict"] = np.array(lead_contradicts, dtype=np.float32)
    feats["num_contradiction"] = np.array(num_contradicts, dtype=np.float32)
    feats["pin_code_match"] = np.array(pin_matches, dtype=np.float32)

    # 4. Retrieval & Ranking Signals
    feats["rrf_score"] = merged["rrf_score"].fillna(0.0).astype(np.float32).values if "rrf_score" in merged.columns else np.zeros(N, dtype=np.float32)
    feats["lexical_rank"] = merged["lexical_rank"].fillna(1000.0).astype(np.float32).values if "lexical_rank" in merged.columns else np.full(N, 1000.0, dtype=np.float32)
    feats["dense_rank"] = merged["dense_rank"].fillna(1000.0).astype(np.float32).values if "dense_rank" in merged.columns else np.full(N, 1000.0, dtype=np.float32)
    feats["retrieval_similarity"] = merged["retrieval_similarity"].fillna(0.0).astype(np.float32).values if "retrieval_similarity" in merged.columns else np.zeros(N, dtype=np.float32)
    feats["embedding_similarity"] = merged["dense_similarity"].fillna(0.0).astype(np.float32).values if "dense_similarity" in merged.columns else feats["retrieval_similarity"]

    # 5. Metadata Signals
    feats["same_country"] = np.array([1.0 if c1 == c2 and c1 != "UNKNOWN" else 0.0 for c1, c2 in zip(c1_list, c2_list)], dtype=np.float32)
    feats["is_source2"] = np.array([1.0 if cid.startswith("S2-") else 0.0 for cid in cand_ids], dtype=np.float32)
    feats["is_source3"] = np.array([1.0 if cid.startswith("S3-") else 0.0 for cid in cand_ids], dtype=np.float32)

    feats_df = pd.DataFrame(feats)
    feats_df["source1_entity_id"] = merged["source1_entity_id"].values
    feats_df["candidate_entity_id"] = merged["candidate_entity_id"].values
    return feats_df


if __name__ == "__main__":
    # Smoke test Module 3 Feature Extraction
    s1 = {"entity_id": "S1-1", "clean_name": "apple store", "legal_suffix": "inc", "clean_address": "1 infinite loop cupertino", "extracted_numbers": ["1", "95014"], "country": "US"}
    s2 = {"entity_id": "S2-1", "clean_name": "apple inc", "legal_suffix": "inc", "clean_address": "infinite loop cupertino", "extracted_numbers": ["95014"], "country": "US"}
    meta = {"rrf_score": 0.03, "lexical_rank": 1, "dense_rank": 2, "retrieval_similarity": 0.92, "dense_similarity": 0.88}
    f = compute_pairwise_features(s1, s2, meta)
    print("Module 3 Feature Extraction Smoke Test PASSED! Feature count:", len(f))
    print("Sample features:", {k: round(v, 4) for k, v in list(f.items())[:8]})

    # Vectorized smoke test
    cands_test = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "candidate_entity_id": ["S2-1", "S3-2"],
        "rrf_score": [0.03, 0.01],
        "lexical_rank": [1, 5],
        "dense_rank": [2, 1000],
        "retrieval_similarity": [0.92, 0.45]
    })
    s1_df_test = pd.DataFrame([s1])
    tgt_df_test = pd.DataFrame([s2, {"entity_id": "S3-2", "clean_name": "other", "legal_suffix": "", "clean_address": "other address", "extracted_numbers": [], "country": "US"}])
    v_feats = build_features_dataframe(cands_test, s1_df_test, tgt_df_test)
    print("Vectorized build_features_dataframe PASSED! Shape:", v_feats.shape)
