"""
Fast Iterative Tuning Loop for Amazon ML Challenge 2026.
Uses Optuna Bayesian Optimization over:
- top_k_candidates: [20, 45]
- tau_singleton: [0.45, 0.85]
- tau_match: [0.50, 0.90]
- delta_margin: [0.03, 0.15]

Optimizes Macro F0.5 on held-out validation set.
Logs every trial to tuning_results.csv with runtime and sub-metrics.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import optuna

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocessing import load_and_preprocess_tsv
from blocking import HybridBlocker
from feature_extraction import build_features_dataframe
from models_gbdt import DynamicHardNegativeGBDT
from evaluate import compute_macro_f05, compute_entity_f05
from postprocessing import apply_two_stage_gating, export_matching_results_tsv
from blocking import format_candidate_pairs_tsv

# Paths
BENCH_DIR = Path("benchmark_dataset")
TRAIN_S1_PATH = BENCH_DIR / "train" / "train_source1.tsv"
TRAIN_GT_PATH = BENCH_DIR / "train" / "train_ground_truth.tsv"
VAL_S1_PATH = BENCH_DIR / "val" / "val_source1.tsv"
VAL_GT_PATH = BENCH_DIR / "val" / "val_ground_truth.tsv"
BENCH_S2_PATH = BENCH_DIR / "benchmark_source2.tsv"
BENCH_S3_PATH = BENCH_DIR / "benchmark_source3.tsv"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = Path("tuning_results.csv")

def load_ground_truth(path):
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1 = parts[0].strip()
            m = parts[1].strip() if len(parts) > 1 else ""
            gt[s1] = set([x.strip() for x in m.split(",") if x.strip()])
    return gt

print("=== [Setup 1/4] Loading Preprocessed Benchmark Splits ===", flush=True)
t0 = time.time()
s1_train = load_and_preprocess_tsv(str(TRAIN_S1_PATH))
gt_train = load_ground_truth(str(TRAIN_GT_PATH))

s1_val = load_and_preprocess_tsv(str(VAL_S1_PATH))
gt_val = load_ground_truth(str(VAL_GT_PATH))

s2_all = load_and_preprocess_tsv(str(BENCH_S2_PATH))
s3_all = load_and_preprocess_tsv(str(BENCH_S3_PATH))
target_df = pd.concat([s2_all, s3_all], ignore_index=True)
print(f"Data loaded in {time.time() - t0:.1f}s: S1_Train={len(s1_train):,}, S1_Val={len(s1_val):,}, Target Pool={len(target_df):,}", flush=True)

# 1. Fit Hybrid Blocker
print("\n=== [Setup 2/4] Indexing Targets & Generating Candidate Pairs ===", flush=True)
t1 = time.time()
blocker = HybridBlocker(top_k=40)
blocker.fit_lexical_index(target_df)

# Train on a clean representative 5,000 S1 sample for super-fast training
sample_train_s1 = s1_train.sample(n=min(5000, len(s1_train)), random_state=42)
train_cands = blocker.retrieve_candidates_for_s1(sample_train_s1, use_dense=False)
train_features = build_features_dataframe(train_cands, sample_train_s1, target_df)

train_s1_ids = train_features["source1_entity_id"].tolist()
train_cand_ids = train_features["candidate_entity_id"].tolist()
train_labels = np.array([1 if c in gt_train.get(s, set()) else 0 for s, c in zip(train_s1_ids, train_cand_ids)])
train_groups = train_features["source1_entity_id"].values
print(f"Candidates generated in {time.time() - t1:.1f}s. Training pairs: {len(train_features):,} (Pos: {(train_labels==1).sum()}, Neg: {(train_labels==0).sum()})", flush=True)

# 2. Train GBDT (with GPU acceleration)
print("\n=== [Setup 3/4] Training GBDT with Hard-Negative Mining (GPU Active) ===", flush=True)
t2 = time.time()
gbdt = DynamicHardNegativeGBDT(n_splits=3, seed=42, use_gpu=True)
gbdt.train_with_hard_negatives(train_features, train_labels, train_groups, max_mining_rounds=2)
print(f"GBDT trained in {time.time() - t2:.1f}s.", flush=True)

# 3. Retrieve Validation Candidates & Score Pairs
print("\n=== [Setup 4/4] Pre-scoring Validation Set for Instant Hyperparameter Evaluation ===", flush=True)
t3 = time.time()
# Retrieve candidate pool (top_k=45) for validation
val_cands_max = blocker.retrieve_candidates_for_s1(s1_val, use_dense=False)
val_features_max = build_features_dataframe(val_cands_max, s1_val, target_df)
val_features_max["predicted_prob"] = gbdt.predict_proba(val_features_max)

all_val_s1_ids = s1_val["entity_id"].tolist()
s1_scored_cache = defaultdict(list)
v_s1 = val_features_max["source1_entity_id"].tolist()
v_cand = val_features_max["candidate_entity_id"].tolist()
v_prob = val_features_max["predicted_prob"].tolist()
v_rank = val_features_max["lexical_rank"].tolist()
for s, c, p, rk in zip(v_s1, v_cand, v_prob, v_rank):
    s1_scored_cache[s].append((c, p, rk))

print(f"Scored {len(val_features_max):,} validation candidate pairs in {time.time() - t3:.1f}s.", flush=True)

# 4. Optuna Iterative Tuning Loop
results_records = []

def objective(trial):
    t_start = time.time()
    
    top_k = trial.suggest_int("top_k", 20, 45, step=5)
    tau_singleton = trial.suggest_float("tau_singleton", 0.45, 0.85, step=0.05)
    tau_match = trial.suggest_float("tau_match", max(0.50, tau_singleton - 0.05), 0.90, step=0.05)
    delta_margin = trial.suggest_float("delta_margin", 0.03, 0.15, step=0.02)
    
    preds_map = {}
    total_prec = []
    total_rec = []
    singleton_correct = 0
    singleton_total = 0
    
    for s1_id in all_val_s1_ids:
        cands = s1_scored_cache.get(s1_id, [])
        sliced = cands[:top_k]
        
        true_matches = gt_val.get(s1_id, set())
        is_true_sing = (len(true_matches) == 0)
        if is_true_sing:
            singleton_total += 1
            
        if not sliced:
            preds_map[s1_id] = set()
            if is_true_sing:
                singleton_correct += 1
            continue
            
        max_prob = max([p for _, p, _ in sliced])
        
        # Singleton Gate
        if max_prob < tau_singleton:
            preds_map[s1_id] = set()
            if is_true_sing:
                singleton_correct += 1
        else:
            # Match Gate with true relative delta_margin constraint
            accepted = [cid for cid, p, _ in sliced if p >= tau_match and (max_prob - p) <= delta_margin]
            preds_map[s1_id] = set(accepted)
            
        pred_set = preds_map[s1_id]
        if pred_set and true_matches:
            tp = len(pred_set & true_matches)
            total_prec.append(tp / len(pred_set))
            total_rec.append(tp / len(true_matches))
        elif pred_set and not true_matches:
            total_prec.append(0.0)
            total_rec.append(0.0)
        elif not pred_set and true_matches:
            total_prec.append(0.0)
            total_rec.append(0.0)
            
    macro_f05 = compute_macro_f05(gt_val, preds_map)
    macro_p = float(np.mean(total_prec)) if total_prec else 0.0
    macro_r = float(np.mean(total_rec)) if total_rec else 0.0
    sing_acc = (singleton_correct / singleton_total) if singleton_total else 1.0
    runtime = time.time() - t_start
    
    row = {
        "trial": trial.number,
        "top_k": top_k,
        "tau_singleton": round(tau_singleton, 3),
        "tau_match": round(tau_match, 3),
        "delta_margin": round(delta_margin, 3),
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "singleton_acc": round(sing_acc, 4),
        "runtime_sec": round(runtime, 2)
    }
    results_records.append(row)
    pd.DataFrame(results_records).to_csv(RESULTS_CSV, index=False)
    
    print(f"[Trial {trial.number:02d}] F0.5 = {macro_f05:.4f} | Prec = {macro_p:.4f} | Rec = {macro_r:.4f} | SingAcc = {sing_acc:.3f} | top_k={top_k}, tau_sing={tau_singleton:.2f}, tau_match={tau_match:.2f}, delta_margin={delta_margin:.2f} ({runtime:.2f}s)", flush=True)
    return macro_f05

print("\n" + "=" * 70, flush=True)
print("=== STARTING OPTUNA ITERATIVE TUNING STUDY (20 Rounds / Early Stopping 5) ===", flush=True)
print("=" * 70, flush=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")

class EarlyStoppingCallback:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = -float("inf")
        self.no_improve = 0

    def __call__(self, study, trial):
        if study.best_value > self.best_score + 1e-4:
            self.best_score = study.best_value
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"\n[Early Stopping] No improvement for {self.patience} consecutive trials. Stopping study.", flush=True)
                study.stop()

study.optimize(objective, n_trials=20, callbacks=[EarlyStoppingCallback(patience=5)])

print("\n" + "=" * 70, flush=True)
print(f"[SUCCESS] TUNING STUDY COMPLETE! Best Trial: #{study.best_trial.number}", flush=True)
print(f"Best Macro F0.5 Score: {study.best_value:.4f}", flush=True)
print("Optimal Parameters:", flush=True)
for k, v in study.best_params.items():
    print(f"  {k}: {v}", flush=True)
print("=" * 70, flush=True)

# Generate Validation Output Files with Best Parameters
best_params = study.best_params
best_top_k = best_params["top_k"]
best_tau_sing = best_params["tau_singleton"]
best_tau_match = best_params["tau_match"]
best_delta_margin = best_params.get("delta_margin", 0.08)

final_preds = {}
cand_pairs_map = {}
for s1_id in all_val_s1_ids:
    cands = s1_scored_cache.get(s1_id, [])[:best_top_k]
    cand_pairs_map[s1_id] = [cid for cid, _, _ in cands]
    
    if not cands:
        final_preds[s1_id] = []
        continue
    max_p = max([p for _, p, _ in cands])
    if max_p < best_tau_sing:
        final_preds[s1_id] = []
    else:
        final_preds[s1_id] = [cid for cid, p, _ in cands if p >= best_tau_match and (max_p - p) <= best_delta_margin]

# Export outputs
matching_out = OUTPUT_DIR / "matching_results.tsv"
candidate_out = OUTPUT_DIR / "candidate_pairs.tsv"
export_matching_results_tsv(final_preds, all_val_s1_ids, str(matching_out))
format_candidate_pairs_tsv(val_cands_max, all_val_s1_ids, str(candidate_out))

print(f"\nOutputs written to {matching_out} and {candidate_out}.", flush=True)
