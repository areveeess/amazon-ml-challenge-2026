"""
Module 3 (Part 2): Dynamic Hard-Negative Mining & GBDT Training Engine.
Implements:
- 5-Fold Stratified Group-K-Fold (grouped by Source 1 entity)
- Dynamic Hard-Negative Mining loop (active negative retraining)
- LightGBM pairwise classifier with custom F0.5 early stopping
- Inference probability generator P(match)
"""

import os
from typing import Dict, List, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score, log_loss
import lightgbm as lgb
from config import ModelConfig


def f05_eval(preds, train_data):
    """Custom Macro F0.5 evaluation metric for LightGBM early stopping.
    Directly reuses evaluate.py's compute_macro_f05:
    - F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    - Computed PER Source-1 entity, then macro-averaged across all Source-1 entities.
    - Singletons count: an entity with no true matches scores 1.0 if empty list predicted,
      0.0 if any match predicted for it.
    """
    labels = train_data.get_label()
    loss = log_loss(labels, preds)

    if hasattr(train_data, "s1_ids") and hasattr(train_data, "cand_ids"):
        from evaluate import compute_macro_f05
        from collections import defaultdict
        s1_ids = train_data.s1_ids
        cand_ids = train_data.cand_ids

        # Build ground truth map from validation labels
        gt_map = defaultdict(set)
        for s1, cid, y in zip(s1_ids, cand_ids, labels):
            if y == 1:
                gt_map[s1].add(cid)
        for s1 in s1_ids:
            if s1 not in gt_map:
                gt_map[s1] = set()

        # Build predictions map (threshold 0.50)
        pred_map = defaultdict(set)
        for s1, cid, p in zip(s1_ids, cand_ids, preds):
            if p >= 0.50:
                pred_map[s1].add(cid)
        for s1 in s1_ids:
            if s1 not in pred_map:
                pred_map[s1] = set()

        score = compute_macro_f05(gt_map, pred_map)
    else:
        binary_preds = (preds >= 0.5).astype(int)
        score = fbeta_score(labels, binary_preds, beta=0.5, zero_division=0.0)

    return [('macro_f05', score, True), ('binary_logloss', loss, False)]



class DynamicHardNegativeGBDT:
    def __init__(self, n_splits: int = ModelConfig.n_splits, seed: int = ModelConfig.seed, use_gpu: bool = None):
        self.n_splits = n_splits
        self.seed = seed
        self.models = []
        self.feature_cols = []
        if use_gpu is None:
            self.use_gpu = getattr(ModelConfig, "use_gpu", True)
        else:
            self.use_gpu = use_gpu

        if self.use_gpu:
            try:
                import catboost as cb
                _test_m = cb.CatBoostClassifier(iterations=1, task_type="GPU", verbose=False)
                _test_m.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0, 1]))
                self.gpu_ready = True
                print("[Module 3] GPU Engine ACTIVE: CatBoost GPU on NVIDIA CUDA Device!")
            except Exception as e:
                print(f"[Module 3 Warning] GPU initialization failed ({e}), falling back to LightGBM CPU.")
                self.use_gpu = False
                self.gpu_ready = False
        else:
            self.gpu_ready = False

    def train_with_hard_negatives(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        groups: np.ndarray,
        lgb_params: dict = None,
        max_mining_rounds: int = None,
        hard_neg_percentile: float = None,
        max_sample_weight: float = None,
    ) -> Tuple[List[Any], np.ndarray]:
        """
        Trains GBDT (CatBoost GPU if use_gpu=True, else LightGBM CPU) using Iterative Hard-Negative Mining:
        1. Train baseline models on initial candidates + hard negatives.
        2. Identify False Positives using percentile-based threshold on
           negative-score distribution (not a hardcoded cutoff).
        3. Re-weight hard negatives in subsequent rounds (with capped escalation).

        Returns:
            (list_of_fold_models, oof_predictions) — OOF predictions are aligned
            with the rows of features_df and are genuinely out-of-fold.
        """
        if max_mining_rounds is None:
            max_mining_rounds = ModelConfig.max_hard_neg_mining_rounds
        if hard_neg_percentile is None:
            hard_neg_percentile = ModelConfig.hard_neg_percentile
        if max_sample_weight is None:
            max_sample_weight = ModelConfig.max_sample_weight

        ignore_cols = {"source1_entity_id", "candidate_entity_id", "label", "group"}
        self.feature_cols = [c for c in features_df.columns if c not in ignore_cols]

        X = features_df[self.feature_cols].values
        y = labels

        # Build params: use ModelConfig.lgb_params as single source of truth for LightGBM
        if lgb_params is None:
            params = ModelConfig.lgb_params.copy()
        else:
            params = lgb_params.copy()
        params["metric"] = "None"
        num_boost_round = int(params.pop("n_estimators", 800))

        n_unique_groups = len(np.unique(groups))
        actual_splits = max(2, min(self.n_splits, n_unique_groups))
        gkf = GroupKFold(n_splits=actual_splits)

        current_sample_weights = np.ones(len(y), dtype=np.float32)
        engine_name = "CatBoost GPU" if self.use_gpu else "LightGBM CPU"
        print(f"[Module 3] Initial training size ({engine_name}): {len(y)} pairs "
              f"(Positives: {(y == 1).sum()}, Negatives: {(y == 0).sum()})")

        oof_preds = np.zeros(len(y))

        for round_idx in range(1, max_mining_rounds + 1):
            print(f"\n--- [Module 3] Hard-Negative Mining Round {round_idx}/{max_mining_rounds} ({engine_name}) ---")
            fold_models = []
            oof_preds = np.zeros(len(y))

            for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
                X_train, y_train = X[train_idx], y[train_idx]
                w_train = current_sample_weights[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                if self.use_gpu:
                    import catboost as cb
                    train_pool = cb.Pool(data=X_train, label=y_train, weight=w_train)
                    val_pool = cb.Pool(data=X_val, label=y_val)

                    cb_params = getattr(ModelConfig, "catboost_params", {}).copy()
                    cb_params["random_seed"] = self.seed + fold

                    model = cb.CatBoostClassifier(**cb_params)
                    model.fit(train_pool, eval_set=val_pool, verbose=False)

                    val_preds = model.predict_proba(val_pool)[:, 1]
                    oof_preds[val_idx] = val_preds
                    fold_models.append(model)
                else:
                    fold_params = params.copy()
                    fold_params["random_state"] = self.seed + fold

                    trn_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
                    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
                    val_data.s1_ids = groups[val_idx]
                    val_data.cand_ids = features_df["candidate_entity_id"].values[val_idx]

                    model = lgb.train(
                        fold_params,
                        trn_data,
                        num_boost_round=num_boost_round,
                        valid_sets=[val_data],
                        feval=f05_eval,
                        callbacks=[lgb.early_stopping(stopping_rounds=40, first_metric_only=True, verbose=False)]
                    )

                    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
                    oof_preds[val_idx] = val_preds
                    fold_models.append(model)

            self.models = fold_models

            # Percentile-based hard negative mining (not a fixed threshold)
            neg_scores = oof_preds[y == 0]
            if len(neg_scores) > 0:
                threshold = np.percentile(neg_scores, hard_neg_percentile)
            else:
                threshold = 0.5
            hard_fp_mask = (y == 0) & (oof_preds > threshold)
            n_mined = hard_fp_mask.sum()
            print(f"[Module 3 Round {round_idx}] Mined {n_mined} hard False Positives "
                  f"(top {100 - hard_neg_percentile:.0f}% of negatives, threshold={threshold:.3f})")

            if n_mined == 0 or round_idx == max_mining_rounds:
                break

            # Upweight hard negatives with capped escalation
            new_weights = current_sample_weights[hard_fp_mask] * 2.5
            current_sample_weights[hard_fp_mask] = np.minimum(new_weights, max_sample_weight)

        return self.models, oof_preds

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Ensemble average across all trained folds."""
        X = features_df[self.feature_cols].values
        fold_preds = np.zeros(len(X))
        for model in self.models:
            if hasattr(model, "predict_proba"):
                fold_preds += model.predict_proba(X)[:, 1] / len(self.models)
            else:
                fold_preds += model.predict(X, num_iteration=model.best_iteration) / len(self.models)
        return fold_preds

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump({"models": self.models, "feature_cols": self.feature_cols, "use_gpu": self.use_gpu}, os.path.join(output_dir, "gbdt_ensemble.pkl"))
        print(f"[Module 3] Saved GBDT models to {output_dir}")

    def load(self, model_path: str):
        data = joblib.load(model_path)
        self.models = data["models"]
        self.feature_cols = data["feature_cols"]
        self.use_gpu = data.get("use_gpu", False)

if __name__ == "__main__":
    # Smoke test Module 3 Model
    feats = pd.DataFrame({
        "name_lev_ratio": np.random.uniform(0, 1, 50),
        "name_jaro_winkler": np.random.uniform(0, 1, 50),
        "addr_word_jaccard": np.random.uniform(0, 1, 50),
        "rrf_score": np.random.uniform(0, 0.05, 50)
    })
    labels = np.random.binomial(1, 0.2, 50)
    groups = np.repeat(np.arange(10), 5)

    trainer = DynamicHardNegativeGBDT(n_splits=3)
    _, oof = trainer.train_with_hard_negatives(feats, labels, groups, max_mining_rounds=2)
    preds = trainer.predict_proba(feats)
    print("Module 3 GBDT Smoke Test PASSED! Sample preds:", np.round(preds[:5], 4))
    print("OOF preds sample:", np.round(oof[:5], 4))
