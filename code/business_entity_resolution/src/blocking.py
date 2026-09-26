"""
Module 2: High-Recall Hybrid Blocking & Candidate Generation Engine.
Implements:
- Lexical Channel: TF-IDF with character 3-grams and token n-grams
- Dense Semantic Channel: Multilingual Bi-Encoder (BGE-M3 / Multilingual-E5) with lazy device detection
- Reciprocal Rank Fusion (RRF) to merge lexical and dense ranks
- Country Partitioning with pre-fitted single global fallback index (unseen countries)
- Memory-safe sparse top-K retrieval without densifying (prevents OOM at 1.7M scale)
- Vectorized formatter for output/candidate_pairs.tsv
"""

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def _sparse_topk(sim_sparse: sp.csr_matrix, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts top-k column indices and scores from each row of a CSR matrix
    WITHOUT ever densifying the matrix (prevents OOM on large target pools).

    Returns a list of (top_cols, top_scores) for each row.
    """
    results = []
    n_rows = sim_sparse.shape[0]
    indptr = sim_sparse.indptr
    indices = sim_sparse.indices
    data = sim_sparse.data

    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i + 1]
        row_vals = data[start:end]
        row_cols = indices[start:end]
        n_vals = len(row_vals)

        if n_vals == 0:
            results.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
            continue

        if n_vals <= k:
            order = np.argsort(-row_vals)
            results.append((row_cols[order], row_vals[order]))
        else:
            part = np.argpartition(-row_vals, k)[:k]
            order = part[np.argsort(-row_vals[part])]
            results.append((row_cols[order], row_vals[order]))

    return results


class HybridBlocker:
    def __init__(self, top_k: int = 40, rrf_k: int = 60):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.target_df = None
        self.country_indices = defaultdict(list)
        self.char_vectorizers = {}
        self.word_vectorizers = {}
        self.char_matrices = {}
        self.word_matrices = {}

        # Global fallback indices (pre-fitted once during fit_lexical_index)
        self.global_char_vec = None
        self.global_word_vec = None
        self.global_char_mat = None
        self.global_word_mat = None
        self.global_target_indices = None

        # Dense model state
        self.dense_model = None

    def _init_dense_model(self, model_name: str = "BAAI/bge-m3"):
        """Lazy load multilingual sentence transformer model if available."""
        if self.dense_model is None:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.dense_model = SentenceTransformer(model_name, device=device)
                print(f"[Module 2] Loaded multilingual dense model: {model_name} on {device}")
            except Exception as e:
                print(f"[Module 2] Dense model unavailable ({e}). Using pure dual-TFIDF lexical fusion.")
                self.dense_model = None

    def fit_lexical_index(self, target_df: pd.DataFrame):
        """
        Builds dual TF-IDF index (Word + Char 3-gram) on Target records (S2 + S3).
        Partitioned by country to maximize speed and precision.
        Also pre-fits a single global fallback index across ALL targets once for unseen countries.
        """
        self.target_df = target_df.reset_index(drop=True)

        # Handle NaN or missing clean_text
        clean_texts = self.target_df["clean_text"].fillna("").astype(str)
        empty_mask = clean_texts.str.strip().eq("") | clean_texts.str.strip().eq("|  |")
        if empty_mask.any():
            print(f"[Module 2 Warning] {empty_mask.sum()} target records have empty clean_text.")

        countries = self.target_df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        self.country_indices = defaultdict(list)
        for idx, country in enumerate(countries):
            self.country_indices[country].append(idx)

        self.char_vectorizers = {}
        self.word_vectorizers = {}
        self.char_matrices = {}
        self.word_matrices = {}

        for country, indices in self.country_indices.items():
            sub_texts = clean_texts.iloc[indices].tolist()

            # Char 3-gram vectorizer
            char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, max_features=40000)
            char_mat = char_vec.fit_transform(sub_texts)

            # Word vectorizer
            word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=30000)
            word_mat = word_vec.fit_transform(sub_texts)

            self.char_vectorizers[country] = char_vec
            self.word_vectorizers[country] = word_vec
            self.char_matrices[country] = char_mat
            self.word_matrices[country] = word_mat

        # Pre-fit a single global fallback index once for unseen countries (e.g. France or mismatches)
        all_texts = clean_texts.tolist()
        self.global_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_features=50000)
        self.global_char_mat = self.global_char_vec.fit_transform(all_texts)
        self.global_word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=50000)
        self.global_word_mat = self.global_word_vec.fit_transform(all_texts)
        self.global_target_indices = np.arange(len(self.target_df))

    def retrieve_candidates_for_s1(self, s1_df: pd.DataFrame, use_dense: bool = False) -> pd.DataFrame:
        """
        Retrieves top-K candidate pairs for each S1 entity using RRF.
        Memory-safe: uses non-densifying sparse top-k.
        Returns a DataFrame with columns:
        [source1_entity_id, candidate_entity_id, rrf_score, lexical_rank, dense_rank, retrieval_similarity]
        """
        if use_dense:
            self._init_dense_model()

        target_entity_ids = self.target_df["entity_id"].values
        candidate_rows = []

        # Reset index of s1_df for clean indexing
        s1_df = s1_df.reset_index(drop=True)

        # Check for empty clean_text in S1 and flag them
        s1_clean_texts = s1_df["clean_text"].fillna("").astype(str)
        empty_s1_mask = s1_clean_texts.str.strip().eq("") | s1_clean_texts.str.strip().eq("|  |")
        if empty_s1_mask.any():
            empty_ids = s1_df.loc[empty_s1_mask, "entity_id"].tolist()
            print(f"[Module 2 Warning] {len(empty_ids)} S1 entities have empty business_name and business_address (e.g. {empty_ids[:3]}).")

        # Group S1 queries by country
        s1_countries = s1_df["country"].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        for country, s1_sub in s1_df.groupby(s1_countries):

            if country not in self.country_indices or len(self.country_indices[country]) == 0:
                # Unmatched country fallback: use pre-fitted global index (NO re-fitting)
                target_sub_indices = self.global_target_indices
                char_vec = self.global_char_vec
                char_mat = self.global_char_mat
                word_vec = self.global_word_vec
                word_mat = self.global_word_mat
            else:
                target_sub_indices = np.array(self.country_indices[country], dtype=np.int64)
                char_vec = self.char_vectorizers[country]
                char_mat = self.char_matrices[country]
                word_vec = self.word_vectorizers[country]
                word_mat = self.word_matrices[country]

            s1_sub_texts = s1_sub["clean_text"].fillna("").astype(str).tolist()
            s1_ids = s1_sub["entity_id"].astype(str).tolist()

            batch_size = 500
            num_targets = len(target_sub_indices)
            k_retrieval = min(self.top_k * 2, num_targets)

            # Pre-compute target dense embeddings if dense channel is active
            target_dense_emb = None
            if use_dense and self.dense_model is not None:
                sub_target_texts = self.target_df.iloc[target_sub_indices]["clean_text"].tolist()
                target_dense_emb = self.dense_model.encode(
                    sub_target_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
                )

            for b_start in range(0, len(s1_ids), batch_size):
                b_end = min(b_start + batch_size, len(s1_ids))
                b_texts = s1_sub_texts[b_start:b_end]
                b_ids = s1_ids[b_start:b_end]

                # Sparse lexical dot products — strictly CSR matrices, no densification
                b_char = char_vec.transform(b_texts)
                b_word = word_vec.transform(b_texts)
                sim_char = b_char.dot(char_mat.T)
                sim_word = b_word.dot(word_mat.T)
                sim_lex = 0.5 * sim_char + 0.5 * sim_word

                # Non-densifying top-K extraction per query
                lex_topk_batch = _sparse_topk(sim_lex, k_retrieval)

                # Optional dense retrieval for this batch
                dense_topk_batch = None
                if target_dense_emb is not None:
                    b_dense_emb = self.dense_model.encode(
                        b_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
                    )
                    b_dense_sim = np.dot(b_dense_emb, target_dense_emb.T)
                    dense_topk_batch = []
                    for row_sim in b_dense_sim:
                        if len(row_sim) <= k_retrieval:
                            order = np.argsort(-row_sim)
                        else:
                            part = np.argpartition(-row_sim, k_retrieval)[:k_retrieval]
                            order = part[np.argsort(-row_sim[part])]
                        dense_topk_batch.append((order, row_sim[order]))

                for i, s1_id in enumerate(b_ids):
                    top_lex_cols, top_lex_vals = lex_topk_batch[i]

                    # Build map of candidate_id -> (rank, sim)
                    lex_cands = {}
                    for rank, (col_idx, score) in enumerate(zip(top_lex_cols, top_lex_vals)):
                        if score <= 0:
                            continue
                        global_target_idx = target_sub_indices[col_idx]
                        cid = target_entity_ids[global_target_idx]
                        lex_cands[cid] = (rank + 1, float(score))

                    dense_cands = {}
                    if dense_topk_batch is not None:
                        top_dense_cols, top_dense_vals = dense_topk_batch[i]
                        for rank, (col_idx, score) in enumerate(zip(top_dense_cols, top_dense_vals)):
                            if score <= 0:
                                continue
                            global_target_idx = target_sub_indices[col_idx]
                            cid = target_entity_ids[global_target_idx]
                            dense_cands[cid] = (rank + 1, float(score))

                    # Union of candidates from lexical and dense channels
                    all_cand_ids = set(lex_cands.keys()) | set(dense_cands.keys())
                    if not all_cand_ids:
                        continue

                    # RRF fusion
                    fused = []
                    for cid in all_cand_ids:
                        l_rank, l_sim = lex_cands.get(cid, (1000, 0.0))
                        d_rank, d_sim = dense_cands.get(cid, (1000, 0.0))

                        rrf = 0.0
                        if cid in lex_cands:
                            rrf += 1.0 / (self.rrf_k + l_rank)
                        if cid in dense_cands:
                            rrf += 1.0 / (self.rrf_k + d_rank)

                        best_sim = l_sim if l_sim > 0 else d_sim
                        fused.append((cid, rrf, l_rank, d_rank, best_sim))

                    # Sort by RRF score descending and take top_k
                    fused.sort(key=lambda x: x[1], reverse=True)
                    for cid, rrf, l_rank, d_rank, sim in fused[:self.top_k]:
                        candidate_rows.append({
                            "source1_entity_id": s1_id,
                            "candidate_entity_id": cid,
                            "rrf_score": rrf,
                            "lexical_rank": l_rank,
                            "dense_rank": d_rank,
                            "retrieval_similarity": sim
                        })

        return pd.DataFrame(candidate_rows)


def format_candidate_pairs_tsv(candidates_df: pd.DataFrame, all_s1_ids: List[str], output_path: str):
    """
    Exports candidates in exact competition format:
    source1_entity_id\tcandidate_entity_ids (comma-separated list, empty for singletons)
    Guarantees every S1 entity appears exactly once.
    Vectorized aggregation using groupby for 24M-row scalability.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not candidates_df.empty:
        # Deduplicate preserving order
        dedup_df = candidates_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
        cand_series = (
            dedup_df.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
            .agg(",".join)
        )
        cand_map = cand_series.to_dict()
    else:
        cand_map = {}

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands_str = cand_map.get(str(s1_id), "")
            f.write(f"{s1_id}\t{cands_str}\n")

    print(f"[Module 2] Successfully wrote candidate pairs to: {output_path}")


if __name__ == "__main__":
    # Smoke test Module 2
    from preprocessing import preprocess_dataframe

    s1_df = preprocess_dataframe(pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": ["Google LLC", "Boulangerie Parisienne", None],
        "business_address": ["1600 Amphitheatre Pkwy, Mountain View, CA", "10 Rue de Rivoli, Paris", None],
        "country": ["US", "France", "US"]
    }))

    tgt_df = preprocess_dataframe(pd.DataFrame({
        "entity_id": ["S2-101", "S2-102", "S3-201"],
        "business_name": ["Google", "Boulangerie Paris", "Apple Store"],
        "business_address": ["Amphitheatre Parkway, Mountain View", "Rivoli, Paris 75001", "1 Infinite Loop"],
        "country": ["US", "India", "US"]
    }))

    blocker = HybridBlocker(top_k=5)
    blocker.fit_lexical_index(tgt_df)
    cands = blocker.retrieve_candidates_for_s1(s1_df, use_dense=False)
    print("Module 2 Smoke Test PASSED!")
    print(cands)

    format_candidate_pairs_tsv(cands, ["S1-001", "S1-002", "S1-003"], "smoke_test_cands.tsv")
    with open("smoke_test_cands.tsv") as f:
        print(f.read())
    os.remove("smoke_test_cands.tsv")
