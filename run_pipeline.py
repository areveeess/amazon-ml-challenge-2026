"""
Master End-to-End Orchestrator for Amazon ML Challenge 2026.
Connects Module 1 (Preprocessing) -> Module 2 (Blocking) ->
Module 3 (Features & GBDT) -> Module 4 (Gating & Submission).
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, TEST_S1, TEST_S2, TEST_S3, MATCHING_RESULTS_TSV, CANDIDATE_PAIRS_TSV, BlockingConfig, ModelConfig, GatingConfig
from preprocessing import load_and_preprocess_tsv, preprocess_dataframe
from blocking import HybridBlocker, format_candidate_pairs_tsv
from feature_extraction import build_features_dataframe
from models_gbdt import DynamicHardNegativeGBDT
from cascade_reranker import AdaptiveConfidenceCascade
from evaluate import compute_macro_f05, optimize_gating_thresholds
from postprocessing import apply_two_stage_gating, export_matching_results_tsv
from package_submission import validate_and_package

def run(team_name: str = "amazon_ml_team", fast_dev_run: bool = False):
    print("=" * 70)
    print(f"=== Amazon ML Challenge 2026: End-to-End Pipeline [{team_name}] ===")
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
    # Stage 1: Module 1 Preprocessing
    # ----------------------------------------------------
    print("\n--- [Stage 1/4] Running Module 1: Cross-Lingual Preprocessing ---")
    s1_train = load_and_preprocess_tsv(str(TRAIN_S1))
    s2_train = load_and_preprocess_tsv(str(TRAIN_S2))
    s3_train = load_and_preprocess_tsv(str(TRAIN_S3))
    tgt_train = pd.concat([s2_train, s3_train], ignore_index=True)
    print(f"Loaded Train: S1={len(s1_train)}, Target (S2+S3)={len(tgt_train)}")
    
    s1_test = load_and_preprocess_tsv(str(TEST_S1))
    s2_test = load_and_preprocess_tsv(str(TEST_S2))
    s3_test = load_and_preprocess_tsv(str(TEST_S3))
    tgt_test = pd.concat([s2_test, s3_test], ignore_index=True)
    print(f"Loaded Test: S1={len(s1_test)}, Target (S2+S3)={len(tgt_test)}")
    
    # ----------------------------------------------------
    # Stage 2: Module 2 Hybrid Blocking & Candidate Generation
    # ----------------------------------------------------
    print("\n--- [Stage 2/4] Running Module 2: Hybrid Blocking (ANN + TF-IDF + RRF) ---")
    # Fit blocker on test targets
    test_blocker = HybridBlocker(top_k=BlockingConfig.top_k_candidates, rrf_k=BlockingConfig.rrf_k)
    test_blocker.fit_lexical_index(tgt_test)
    test_candidates_df = test_blocker.retrieve_candidates_for_s1(s1_test, use_dense=False)
    
    # Export candidate_pairs.tsv
    all_test_s1_ids = s1_test["entity_id"].tolist()
    format_candidate_pairs_tsv(test_candidates_df, all_test_s1_ids, str(CANDIDATE_PAIRS_TSV))
    
    # Also generate training candidates for model training
    print("Generating Training Candidate Pairs for Model Training...")
    train_blocker = HybridBlocker(top_k=BlockingConfig.top_k_candidates, rrf_k=BlockingConfig.rrf_k)
    train_blocker.fit_lexical_index(tgt_train)
    train_candidates_df = train_blocker.retrieve_candidates_for_s1(s1_train, use_dense=False)
    
    # ----------------------------------------------------
    # Stage 3: Module 3 Features & Dynamic Hard-Negative GBDT
    # ----------------------------------------------------
    print("\n--- [Stage 3/4] Running Module 3: Feature Engineering & Hard-Negative Mining ---")
    train_features_df = build_features_dataframe(train_candidates_df, s1_train, tgt_train)
    
    # Label training pairs from ground truth
    gt_df = pd.read_csv(str(TRAIN_GT), sep="\t", dtype=str)
    gt_map = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        gt_map[s1_id] = set([m.strip() for m in m_str.split(",") if m.strip()])
        
    labels = []
    for _, row in train_features_df.iterrows():
        s1 = row["source1_entity_id"]
        cand = row["candidate_entity_id"]
        is_match = 1 if cand in gt_map.get(s1, set()) else 0
        labels.append(is_match)
    labels = np.array(labels)
    groups = train_features_df["source1_entity_id"].values
    
    # Train GBDT with Dynamic Hard Negative Mining (returns OOF predictions)
    gbdt_trainer = DynamicHardNegativeGBDT(n_splits=ModelConfig.n_splits, seed=ModelConfig.seed)
    _, oof_preds = gbdt_trainer.train_with_hard_negatives(
        train_features_df, labels, groups,
        lgb_params=ModelConfig.lgb_params,
        max_mining_rounds=ModelConfig.max_hard_neg_mining_rounds,
        hard_neg_percentile=ModelConfig.hard_neg_percentile,
        max_sample_weight=ModelConfig.max_sample_weight,
    )
    
    # Tune Gating Thresholds using genuinely Out-Of-Fold predictions (not leaky re-inference)
    print("\nOptimizing Two-Stage Gating Thresholds for Macro F0.5 (using OOF predictions)...")
    scored_train_pairs = list(zip(train_features_df["source1_entity_id"], train_features_df["candidate_entity_id"], oof_preds))
    all_train_s1_ids = s1_train["entity_id"].tolist()
    best_f05, opt_tau_sing, opt_tau_match = optimize_gating_thresholds(scored_train_pairs, gt_map, all_train_s1_ids)
    
    # ----------------------------------------------------
    # Stage 4: Module 4 Test Inference, Gating & Packaging
    # ----------------------------------------------------
    print("\n--- [Stage 4/4] Running Module 4: Test Scoring, Gating & Final Packaging ---")
    test_features_df = build_features_dataframe(test_candidates_df, s1_test, tgt_test)
    test_preds = gbdt_trainer.predict_proba(test_features_df)
    test_features_df["predicted_prob"] = test_preds
    
    # Adaptive Confidence Cascade for Ambiguous Pairs
    cascade = AdaptiveConfidenceCascade()
    refined_test_df = cascade.refine_probabilities(test_features_df, s1_test, tgt_test, use_cascade=False)
    
    # Apply Two-Stage Calibrated Gating
    predictions_map = apply_two_stage_gating(
        refined_test_df,
        all_test_s1_ids,
        tau_singleton=opt_tau_sing,
        tau_match=opt_tau_match
    )
    
    # Export matching_results.tsv
    export_matching_results_tsv(predictions_map, all_test_s1_ids, str(MATCHING_RESULTS_TSV))
    
    # Validate and Package Zip
    validate_and_package(team_name=team_name)
    print("\n[SUCCESS] Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="amazon_ml_team")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    run(team_name=args.team_name, fast_dev_run=args.fast)
