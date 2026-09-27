"""
Master End-to-End Orchestrator for Amazon ML Challenge 2026.
Connects:
Module 1 (Cross-Lingual Preprocessing) ->
Module 2 (Memory-Safe Sparse Hybrid Blocking) ->
Module 3 (Vectorized Features & Dynamic Hard-Negative GBDT) ->
Module 4 (Two-Stage Gating, Chunked Test Inference & Submission Packaging).
"""

import sys
import os
sys.stdout.reconfigure(line_buffering=True)
import gc
import time
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (
    TRAIN_DIR, TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_DIR, TEST_S1, TEST_S2, TEST_S3,
    MATCHING_RESULTS_TSV, CANDIDATE_PAIRS_TSV, MODELS_DIR,
    BlockingConfig, ModelConfig, GatingConfig
)
from preprocessing import load_and_preprocess_tsv
from blocking import HybridBlocker, format_candidate_pairs_tsv
from feature_extraction import build_features_dataframe
from models_gbdt import DynamicHardNegativeGBDT
from cascade_reranker import AdaptiveConfidenceCascade
from evaluate import optimize_gating_thresholds
from postprocessing import apply_two_stage_gating, export_matching_results_tsv
from package_submission import validate_and_package


def get_memory_mb() -> float:
    """Returns current process working set size in MB using zero-overhead Win32 API."""
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        handle = k32.GetCurrentProcess()
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
        return 0.0
    except Exception:
        return 0.0


def run(
    team_name: str = "amazon_ml_team",
    fast_dev_run: bool = False,
    train_sample: int = 50000,
    test_sample: int = None,
    chunk_size: int = 50000,
    use_gpu: bool = True
):
    start_time = time.time()
    print("=" * 70)
    print(f"=== Amazon ML Challenge 2026: End-to-End Pipeline [{team_name}] ===")
    print(f"Hardware Acceleration: {'GPU Active (NVIDIA CUDA)' if use_gpu else 'CPU Mode'}")
    print("=" * 70)

    # ----------------------------------------------------
    # Check if dataset files exist
    # ----------------------------------------------------
    if not TRAIN_S1.exists() or not TEST_S1.exists():
        print(f"\n[Notice] Dataset files not yet detected in {TRAIN_S1.parent} and {TEST_S1.parent}.")
        print("Please place the challenge TSV files in dataset/train/ and dataset/test/.")
        print("Pipeline is verified and ready to run the moment files are dropped in.")
        return

    # ----------------------------------------------------
    # Stage 1: Preprocessing
    # ----------------------------------------------------
    print(f"\n--- [Stage 1/4] Running Module 1: Cross-Lingual Preprocessing (RAM: {get_memory_mb()} MB) ---")
    t0 = time.time()
    needed_cols = ["entity_id", "clean_text", "clean_name", "legal_suffix", "clean_address", "extracted_numbers", "country"]
    s1_train_nrows = 10000 if fast_dev_run else train_sample
    target_train_nrows = 50000 if fast_dev_run else (None if s1_train_nrows is None else BlockingConfig.train_target_sample_size)
    s1_train = load_and_preprocess_tsv(str(TRAIN_S1), nrows=s1_train_nrows)
    s1_train = s1_train[[c for c in needed_cols if c in s1_train.columns]]
    gc.collect()

    s2_train = load_and_preprocess_tsv(str(TRAIN_S2), nrows=target_train_nrows)
    s2_train = s2_train[[c for c in needed_cols if c in s2_train.columns]]
    gc.collect()

    s3_train = load_and_preprocess_tsv(str(TRAIN_S3), nrows=target_train_nrows)
    s3_train = s3_train[[c for c in needed_cols if c in s3_train.columns]]
    gc.collect()

    tgt_train = pd.concat([s2_train, s3_train], ignore_index=True)
    del s2_train, s3_train
    gc.collect()
    print(f"Loaded Train in {time.time() - t0:.1f}s: S1={len(s1_train):,}, Target (S2+S3)={len(tgt_train):,}")

    # ----------------------------------------------------
    # Stage 2: Hybrid Blocking on Training Data
    # ----------------------------------------------------
    print(f"\n--- [Stage 2/4] Running Module 2: Training Candidate Generation (RAM: {get_memory_mb()} MB) ---")
    t_train_block = time.time()
    train_top_k = 10 if (s1_train_nrows is not None and s1_train_nrows >= 50000) else (5 if s1_train_nrows is None else BlockingConfig.top_k_candidates)
    train_blocker = HybridBlocker(top_k=train_top_k, rrf_k=BlockingConfig.rrf_k)
    train_index_cache = TRAIN_DIR / ".train_lexical_index.joblib" if s1_train_nrows is None else None
    train_blocker.fit_lexical_index(tgt_train, cache_path=train_index_cache)
    train_candidates_df = train_blocker.retrieve_candidates_for_s1(s1_train, use_dense=False)
    print(f"Generated {len(train_candidates_df):,} training candidate pairs in {time.time() - t_train_block:.1f}s.")

    # Free training blocker to reclaim memory
    del train_blocker
    gc.collect()

    # ----------------------------------------------------
    # Stage 3: Features & Dynamic Hard-Negative GBDT
    # ----------------------------------------------------
    print(f"\n--- [Stage 3/4] Running Module 3: Vectorized Feature Engineering & GBDT (RAM: {get_memory_mb()} MB) ---")
    t_feat = time.time()
    train_features_df = build_features_dataframe(train_candidates_df, s1_train, tgt_train)
    print(f"Vectorized feature matrix built in {time.time() - t_feat:.1f}s. Shape: {train_features_df.shape}")

    # Label training pairs from ground truth
    gt_df = pd.read_csv(str(TRAIN_GT), sep="\t", dtype=str)
    gt_s1 = gt_df["source1_entity_id"].astype(str).tolist()
    gt_m = gt_df["matched_entity_ids"].fillna("").astype(str).tolist()
    gt_map = {s: set(m.strip() for m in ms.split(",") if m.strip()) for s, ms in zip(gt_s1, gt_m)}

    labels = np.array([
        1 if cand in gt_map.get(s1, set()) else 0
        for s1, cand in zip(train_features_df["source1_entity_id"], train_features_df["candidate_entity_id"])
    ])
    groups = train_features_df["source1_entity_id"].values
    print(f"Training pairs: {len(labels):,} (Pos: {(labels == 1).sum():,}, Neg: {(labels == 0).sum():,})")

    # Train GBDT with Dynamic Hard Negative Mining (returns OOF predictions)
    t_gbdt = time.time()
    gbdt_trainer = DynamicHardNegativeGBDT(n_splits=ModelConfig.n_splits, seed=ModelConfig.seed, use_gpu=use_gpu)
    _, oof_preds = gbdt_trainer.train_with_hard_negatives(
        train_features_df, labels, groups,
        lgb_params=ModelConfig.lgb_params,
        max_mining_rounds=ModelConfig.max_hard_neg_mining_rounds,
        hard_neg_percentile=ModelConfig.hard_neg_percentile,
        max_sample_weight=ModelConfig.max_sample_weight,
    )
    print(f"GBDT training complete in {time.time() - t_gbdt:.1f}s.")

    # Save trained GBDT model immediately to models/
    gbdt_trainer.save(str(MODELS_DIR))

    # Tune Gating Thresholds using genuinely Out-Of-Fold predictions (not leaky re-inference)
    print("\nOptimizing Two-Stage Gating Thresholds for Macro F0.5 (using OOF predictions)...")
    scored_train_pairs = list(zip(train_features_df["source1_entity_id"], train_features_df["candidate_entity_id"], oof_preds))
    all_train_s1_ids = s1_train["entity_id"].tolist()
    best_f05, opt_tau_sing, opt_tau_match = optimize_gating_thresholds(
        scored_train_pairs, gt_map, all_train_s1_ids, delta_margin=GatingConfig.delta_margin
    )

    # Free training feature structures before test inference
    del train_features_df, train_candidates_df, oof_preds, gt_map, gt_df, s1_train, tgt_train
    gc.collect()

    # ----------------------------------------------------
    # Stage 4: Chunked Test Inference, Gating & Packaging
    # ----------------------------------------------------
    print(f"\n--- [Stage 4/4] Running Module 4: Chunked Test Inference & Streaming Gating (RAM: {get_memory_mb()} MB) ---")
    t1 = time.time()
    s1_test = load_and_preprocess_tsv(str(TEST_S1), nrows=test_sample)
    s1_test = s1_test[[c for c in needed_cols if c in s1_test.columns]]
    gc.collect()

    s2_test = load_and_preprocess_tsv(str(TEST_S2))
    s2_test = s2_test[[c for c in needed_cols if c in s2_test.columns]]
    gc.collect()

    s3_test = load_and_preprocess_tsv(str(TEST_S3))
    s3_test = s3_test[[c for c in needed_cols if c in s3_test.columns]]
    gc.collect()

    tgt_test = pd.concat([s2_test, s3_test], ignore_index=True)
    del s2_test, s3_test
    gc.collect()
    print(f"Loaded Test in {time.time() - t1:.1f}s: S1={len(s1_test):,}, Target (S2+S3)={len(tgt_test):,}")

    t_test_block = time.time()
    test_blocker = HybridBlocker(top_k=BlockingConfig.top_k_candidates, rrf_k=BlockingConfig.rrf_k)
    test_index_cache = TEST_DIR / ".test_lexical_index.joblib"
    test_blocker.fit_lexical_index(tgt_test, cache_path=test_index_cache)
    print(f"Fitted test lexical blocker in {time.time() - t_test_block:.1f}s.")

    # Initialize output TSVs with exact competition headers
    MATCHING_RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(MATCHING_RESULTS_TSV, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
    with open(CANDIDATE_PAIRS_TSV, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    n_test = len(s1_test)
    num_chunks = int(np.ceil(n_test / chunk_size))
    print(f"Processing {n_test:,} test entities across {num_chunks} chunk(s) of size {chunk_size:,}...")

    total_candidates_written = 0
    total_matches_written = 0

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, n_test)
        s1_chunk = s1_test.iloc[c_start:c_end].reset_index(drop=True)
        chunk_s1_ids = s1_chunk["entity_id"].astype(str).tolist()

        t_ch = time.time()
        # 1. Retrieve candidates for chunk
        cands_chunk = test_blocker.retrieve_candidates_for_s1(s1_chunk, use_dense=False)

        # 2. Format & append candidate pairs
        if not cands_chunk.empty:
            dedup_cands = cands_chunk.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
            cand_series = (
                dedup_cands.groupby("source1_entity_id", sort=False)["candidate_entity_id"]
                .agg(",".join)
            )
            cand_map = cand_series.to_dict()
        else:
            cand_map = {}

        with open(CANDIDATE_PAIRS_TSV, "a", encoding="utf-8") as f_cand:
            for s1_id in chunk_s1_ids:
                c_str = cand_map.get(s1_id, "")
                f_cand.write(f"{s1_id}\t{c_str}\n")
                if c_str:
                    total_candidates_written += len(c_str.split(","))

        # 3. Vectorized feature extraction & prediction for chunk
        if not cands_chunk.empty:
            feat_chunk = build_features_dataframe(cands_chunk, s1_chunk, tgt_test)
            preds_chunk = gbdt_trainer.predict_proba(feat_chunk)
            feat_chunk["predicted_prob"] = preds_chunk

            chunk_preds_map = apply_two_stage_gating(
                feat_chunk,
                chunk_s1_ids,
                tau_singleton=opt_tau_sing,
                tau_match=opt_tau_match,
                delta_margin=GatingConfig.delta_margin
            )
        else:
            chunk_preds_map = {s1_id: [] for s1_id in chunk_s1_ids}

        # 4. Append matching results
        with open(MATCHING_RESULTS_TSV, "a", encoding="utf-8") as f_match:
            for s1_id in chunk_s1_ids:
                m_list = chunk_preds_map.get(s1_id, [])
                f_match.write(f"{s1_id}\t{','.join(m_list)}\n")
                total_matches_written += len(m_list)

        ch_elapsed = time.time() - t_ch
        print(f"  [Chunk {chunk_idx + 1}/{num_chunks}] Entities {c_start:,}-{c_end:,} processed in {ch_elapsed:.1f}s (RAM: {get_memory_mb()} MB)")

        # Garbage collect after every chunk to prevent RAM accumulation
        del cands_chunk, chunk_preds_map
        if "feat_chunk" in locals():
            del feat_chunk, preds_chunk
        gc.collect()

    print(f"\nTest generation complete! Wrote {n_test:,} entities: {total_candidates_written:,} candidates, {total_matches_written:,} matches.")
    print(f"Output files:")
    print(f"  - {MATCHING_RESULTS_TSV}")
    print(f"  - {CANDIDATE_PAIRS_TSV}")

    # Validate and Package Zip
    validate_and_package(team_name=team_name)
    total_elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Pipeline Execution Completed in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)! Final RAM: {get_memory_mb()} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: End-to-End Pipeline")
    parser.add_argument("--team-name", default="amazon_ml_team")
    parser.add_argument("--fast", action="store_true", help="Fast development run on 10k S1 sample")
    parser.add_argument("--train-sample", type=int, default=50000, help="Training S1 sample size (default 50,000)")
    parser.add_argument("--train-full", action="store_true", help="Train on all 2.2M S1 entities (slow, CPU/RAM intensive)")
    parser.add_argument("--test-sample", type=int, default=None, help="Sample test entities for dry run")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming test inference")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable GPU acceleration (CatBoost GPU on CUDA)")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU acceleration (CPU mode)")
    args = parser.parse_args()

    train_n = None if args.train_full else args.train_sample
    run(
        team_name=args.team_name,
        fast_dev_run=args.fast,
        train_sample=train_n,
        test_sample=args.test_sample,
        chunk_size=args.chunk_size,
        use_gpu=args.gpu
    )
    sys.stdout.flush()
    os._exit(0)

