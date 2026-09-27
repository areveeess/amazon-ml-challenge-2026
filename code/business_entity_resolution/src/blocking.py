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
from pathlib import Path
import joblib
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except (ImportError, Exception):
    HAS_TORCH_CUDA = False


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

    def fit_lexical_index(self, target_df: pd.DataFrame, cache_path: Any = None):
        """
        Builds dual TF-IDF index (Word + Char 3-gram) on Target records (S2 + S3).
        Partitioned by country to maximize speed and precision.
        Also pre-fits a single global fallback index across ALL targets once for unseen countries.
        Supports disk caching to instantly load pre-fitted indexes on repeat runs.
        """
        self.target_df = target_df.reset_index(drop=True)

        if cache_path is not None:
            cache_p = Path(cache_path)
            if cache_p.exists():
                try:
                    print(f"[Module 2 Cache] Loading pre-fitted lexical index from {cache_p.name}...")
                    cached = joblib.load(cache_p)
                    self.country_indices = cached["country_indices"]
                    self.char_vectorizers = cached["char_vectorizers"]
                    self.word_vectorizers = cached["word_vectorizers"]
                    self.char_matrices = cached["char_matrices"]
                    self.word_matrices = cached["word_matrices"]
                    self.global_char_vec = cached.get("global_char_vec", None)
                    self.global_char_mat = cached.get("global_char_mat", None)
                    self.global_word_vec = cached.get("global_word_vec", None)
                    self.global_word_mat = cached.get("global_word_mat", None)
                    self.global_target_indices = cached.get("global_target_indices", None)
                    print(f"[Module 2 Cache] Pre-fitted lexical index loaded in seconds!")
                    return
                except Exception as e:
                    print(f"[Module 2 Warning] Cache load failed ({e}), rebuilding index...")

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
            char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, max_features=40000, dtype=np.float32)
            char_mat = char_vec.fit_transform(sub_texts)

            # Word vectorizer: unigram inverted index for speed and low memory footprint
            word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2, max_features=35000, dtype=np.float32)
            word_mat = word_vec.fit_transform(sub_texts)

            self.char_vectorizers[country] = char_vec
            self.word_vectorizers[country] = word_vec
            self.char_matrices[country] = char_mat
            self.word_matrices[country] = word_mat

        # Unseen countries (e.g. France) search against ALL country partitions merged via RRF
        # No 50k subsampling fallback needed - full target pool is searched!
        self.global_char_vec = None
        self.global_char_mat = None
        self.global_word_vec = None
        self.global_word_mat = None
        self.global_target_indices = None

        if cache_path is not None:
            try:
                cache_p = Path(cache_path)
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                to_cache = {
                    "country_indices": self.country_indices,
                    "char_vectorizers": self.char_vectorizers,
                    "word_vectorizers": self.word_vectorizers,
                    "char_matrices": self.char_matrices,
                    "word_matrices": self.word_matrices,
                    "global_char_vec": self.global_char_vec,
                    "global_char_mat": self.global_char_mat,
                    "global_word_vec": self.global_word_vec,
                    "global_word_mat": self.global_word_mat,
                    "global_target_indices": self.global_target_indices,
                }
                joblib.dump(to_cache, cache_p, compress=1)
                print(f"[Module 2 Cache] Saved pre-fitted lexical index to {cache_p.name}")
            except Exception as e:
                print(f"[Module 2 Warning] Could not save index cache: {e}")

    def _retrieve_batch_from_partition(
        self,
        b_texts: List[str],
        b_ids: List[str],
        target_sub_indices: np.ndarray,
        char_vec: TfidfVectorizer,
        char_mat: Any,
        word_vec: TfidfVectorizer,
        word_mat: Any,
        k_retrieval: int,
        target_entity_ids: np.ndarray,
        target_dense_emb: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Tuple[int, float]]], Optional[List[Dict[str, Tuple[int, float]]]]]:
        """
        Retrieves top candidate IDs for a batch of query texts from a single country partition.
        Returns:
            (lex_cands_batch, dense_cands_batch)
            where each is a list (per query in b_ids) of dict: {candidate_entity_id: (partition_rank, score)}
        """
        # Sparse lexical retrieval:
        # Stage 1: Fast word-level inverted index matching
        b_word = word_vec.transform(b_texts)
        sim_word = b_word.dot(word_mat.T)
        b_char = char_vec.transform(b_texts)

        # Stage 2: Rescore top candidates with character 3-grams
        lex_cands_batch = []
        for i in range(len(b_ids)):
            start = sim_word.indptr[i]
            end = sim_word.indptr[i + 1]
            w_vals = sim_word.data[start:end]
            w_cols = sim_word.indices[start:end]
            n_w = len(w_vals)

            if n_w == 0:
                # Fallback for zero word overlap (e.g. typos): single-row char dot product
                single_char_sim = b_char[i].dot(char_mat.T)
                top_cols, top_vals = _sparse_topk(single_char_sim, k_retrieval)[0]
                q_lex_cands = {}
                for rank, (col_idx, score) in enumerate(zip(top_cols, top_vals)):
                    if score <= 0:
                        continue
                    cid = target_entity_ids[target_sub_indices[col_idx]]
                    q_lex_cands[cid] = (rank + 1, float(score))
                lex_cands_batch.append(q_lex_cands)
                continue

            # Select candidate pool: up to top 60 from word overlap (preserves top-40 while doubling speed)
            pool_k = min(max(self.top_k + 20, 60), n_w)
            if n_w <= pool_k:
                pool_indices = np.arange(n_w)
            else:
                pool_indices = np.argpartition(-w_vals, pool_k)[:pool_k]

            cand_cols = w_cols[pool_indices]
            cand_w_scores = w_vals[pool_indices]

            # Compute exact char 3-gram score on this candidate pool
            sub_char_mat = char_mat[cand_cols]
            sub_char_sim = b_char[i].dot(sub_char_mat.T).toarray()[0]

            # Combined lexical score: 50% word + 50% char
            combined_scores = 0.5 * cand_w_scores + 0.5 * sub_char_sim

            # Rank candidates (GPU tensor topk if available on device)
            n_cands = len(combined_scores)
            k_c = min(k_retrieval, n_cands)
            if n_cands <= k_c:
                sort_order = np.argsort(-combined_scores)
            else:
                part = np.argpartition(-combined_scores, k_c)[:k_c]
                sort_order = part[np.argsort(-combined_scores[part])]

            q_lex_cands = {}
            for rank, idx in enumerate(sort_order):
                score = combined_scores[idx]
                if score <= 0:
                    continue
                col_idx = cand_cols[idx]
                cid = target_entity_ids[target_sub_indices[col_idx]]
                q_lex_cands[cid] = (rank + 1, float(score))

            lex_cands_batch.append(q_lex_cands)

        # Optional dense retrieval for this batch
        dense_cands_batch = None
        if target_dense_emb is not None:
            b_dense_emb = self.dense_model.encode(
                b_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            b_dense_sim = np.dot(b_dense_emb, target_dense_emb.T)
            dense_cands_batch = []
            for row_sim in b_dense_sim:
                if len(row_sim) <= k_retrieval:
                    order = np.argsort(-row_sim)
                else:
                    part = np.argpartition(-row_sim, k_retrieval)[:k_retrieval]
                    order = part[np.argsort(-row_sim[part])]
                q_dense = {}
                for rank, col_idx in enumerate(order[:k_retrieval]):
                    score = row_sim[col_idx]
                    if score <= 0:
                        continue
                    cid = target_entity_ids[target_sub_indices[col_idx]]
                    q_dense[cid] = (rank + 1, float(score))
                dense_cands_batch.append(q_dense)

        return lex_cands_batch, dense_cands_batch

    def retrieve_candidates_for_s1(self, s1_df: pd.DataFrame, use_dense: bool = False) -> pd.DataFrame:
        """
        Retrieves top-K candidate pairs for each S1 entity using RRF.
        Memory-safe: uses non-densifying sparse top-k with safe batching.
        Returns a DataFrame with columns:
        [source1_entity_id, candidate_entity_id, rrf_score, lexical_rank, dense_rank, retrieval_similarity]
        """
        if use_dense:
            self._init_dense_model()

        target_entity_ids = self.target_df["entity_id"].values
        c_s1 = []
        c_cand = []
        c_rrf = []
        c_lrank = []
        c_drank = []
        c_sim = []

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
            # Determine which partitions to search
            if country in self.country_indices and len(self.country_indices[country]) > 0:
                partitions_to_search = [country]
                is_unseen = False
            else:
                # Unseen country (e.g. France when France not in target pool, or country mismatch):
                # Search ALL country partitions across the full target pool and merge via RRF
                partitions_to_search = [c for c in self.country_indices.keys() if len(self.country_indices[c]) > 0]
                is_unseen = True
                total_target_records = sum(len(self.country_indices[c]) for c in partitions_to_search)
                print(f"[Module 2 Fallback] Unseen country '{country}' ({len(s1_sub)} entities) -> Searching ALL {len(partitions_to_search)} country partitions ({total_target_records:,} targets) merged via RRF.")

            if not partitions_to_search:
                continue

            s1_sub_texts = s1_sub["clean_text"].fillna("").astype(str).tolist()
            s1_ids = s1_sub["entity_id"].astype(str).tolist()

            batch_size = 100

            for b_start in range(0, len(s1_ids), batch_size):
                if b_start > 0 and b_start % 50000 == 0:
                    print(f"  [Blocking {country}] {b_start:,}/{len(s1_ids):,} queries processed ({b_start / len(s1_ids):.1%})...", flush=True)
                b_end = min(b_start + batch_size, len(s1_ids))
                b_texts = s1_sub_texts[b_start:b_end]
                b_ids = s1_ids[b_start:b_end]

                if not is_unseen:
                    # Single partition (seen country)
                    part_c = partitions_to_search[0]
                    target_sub_indices = np.array(self.country_indices[part_c], dtype=np.int64)
                    char_vec = self.char_vectorizers[part_c]
                    char_mat = self.char_matrices[part_c]
                    word_vec = self.word_vectorizers[part_c]
                    word_mat = self.word_matrices[part_c]
                    k_retrieval = min(self.top_k * 2, len(target_sub_indices))

                    target_dense_emb = None
                    if use_dense and self.dense_model is not None:
                        sub_target_texts = self.target_df.iloc[target_sub_indices]["clean_text"].tolist()
                        target_dense_emb = self.dense_model.encode(
                            sub_target_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
                        )

                    lex_batch, dense_batch = self._retrieve_batch_from_partition(
                        b_texts, b_ids, target_sub_indices,
                        char_vec, char_mat, word_vec, word_mat,
                        k_retrieval, target_entity_ids, target_dense_emb
                    )

                    for i, s1_id in enumerate(b_ids):
                        lex_cands = lex_batch[i]
                        dense_cands = dense_batch[i] if dense_batch else {}

                        all_cand_ids = set(lex_cands.keys()) | set(dense_cands.keys())
                        if not all_cand_ids:
                            continue

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

                        fused.sort(key=lambda x: (x[1], x[4]), reverse=True)
                        for cid, rrf, l_rank, d_rank, sim in fused[:self.top_k]:
                            c_s1.append(s1_id)
                            c_cand.append(cid)
                            c_rrf.append(rrf)
                            c_lrank.append(l_rank)
                            c_drank.append(d_rank)
                            c_sim.append(sim)

                else:
                    # Unseen country: search ALL partitions and merge via RRF across full target pool
                    batch_partition_lex = [{} for _ in range(len(b_ids))]
                    batch_partition_dense = [{} for _ in range(len(b_ids))] if use_dense else None

                    for part_c in partitions_to_search:
                        target_sub_indices = np.array(self.country_indices[part_c], dtype=np.int64)
                        char_vec = self.char_vectorizers[part_c]
                        char_mat = self.char_matrices[part_c]
                        word_vec = self.word_vectorizers[part_c]
                        word_mat = self.word_matrices[part_c]
                        k_retrieval = min(self.top_k * 2, len(target_sub_indices))

                        target_dense_emb = None
                        if use_dense and self.dense_model is not None:
                            sub_target_texts = self.target_df.iloc[target_sub_indices]["clean_text"].tolist()
                            target_dense_emb = self.dense_model.encode(
                                sub_target_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
                            )

                        part_lex_batch, part_dense_batch = self._retrieve_batch_from_partition(
                            b_texts, b_ids, target_sub_indices,
                            char_vec, char_mat, word_vec, word_mat,
                            k_retrieval, target_entity_ids, target_dense_emb
                        )

                        for i in range(len(b_ids)):
                            batch_partition_lex[i].update(part_lex_batch[i])
                            if use_dense and part_dense_batch:
                                batch_partition_dense[i].update(part_dense_batch[i])

                    # Merge candidates across all partitions via RRF
                    for i, s1_id in enumerate(b_ids):
                        lex_candidates = batch_partition_lex[i]
                        dense_candidates = batch_partition_dense[i] if batch_partition_dense else {}

                        all_cand_ids = set(lex_candidates.keys()) | set(dense_candidates.keys())
                        if not all_cand_ids:
                            continue

                        # Rank lexical candidates across all partitions by similarity
                        sorted_lex = sorted(lex_candidates.items(), key=lambda x: x[1][1], reverse=True)
                        lex_global_ranks = {cid: (rank + 1, score) for rank, (cid, (p_rank, score)) in enumerate(sorted_lex)}

                        if dense_candidates:
                            sorted_dense = sorted(dense_candidates.items(), key=lambda x: x[1][1], reverse=True)
                            dense_global_ranks = {cid: (rank + 1, score) for rank, (cid, (p_rank, score)) in enumerate(sorted_dense)}
                        else:
                            dense_global_ranks = {}

                        fused = []
                        for cid in all_cand_ids:
                            l_rank, l_sim = lex_global_ranks.get(cid, (1000, 0.0))
                            d_rank, d_sim = dense_global_ranks.get(cid, (1000, 0.0))

                            rrf = 0.0
                            if cid in lex_global_ranks:
                                rrf += 1.0 / (self.rrf_k + l_rank)
                            if cid in dense_global_ranks:
                                rrf += 1.0 / (self.rrf_k + d_rank)

                            best_sim = l_sim if l_sim > 0 else d_sim
                            fused.append((cid, rrf, l_rank, d_rank, best_sim))

                        fused.sort(key=lambda x: (x[1], x[4]), reverse=True)
                        for cid, rrf, l_rank, d_rank, sim in fused[:self.top_k]:
                            c_s1.append(s1_id)
                            c_cand.append(cid)
                            c_rrf.append(rrf)
                            c_lrank.append(l_rank)
                            c_drank.append(d_rank)
                            c_sim.append(sim)

        return pd.DataFrame({
            "source1_entity_id": c_s1,
            "candidate_entity_id": c_cand,
            "rrf_score": np.array(c_rrf, dtype=np.float32) if c_rrf else np.empty(0, dtype=np.float32),
            "lexical_rank": np.array(c_lrank, dtype=np.int32) if c_lrank else np.empty(0, dtype=np.int32),
            "dense_rank": np.array(c_drank, dtype=np.int32) if c_drank else np.empty(0, dtype=np.int32),
            "retrieval_similarity": np.array(c_sim, dtype=np.float32) if c_sim else np.empty(0, dtype=np.float32)
        })


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
