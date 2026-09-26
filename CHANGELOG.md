# CHANGELOG — Amazon ML Challenge 2026 Entity Resolution Pipeline

## Group A — Core Scoring Correctness (models_gbdt.py, config.py, run_pipeline.py)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| A.1 | `models_gbdt.py`, `run_pipeline.py` | Return OOF predictions from `train_with_hard_negatives`; feed OOF preds (not leaky re-inferred train preds) into `optimize_gating_thresholds` | Prevents threshold overconfidence (OOF-calibrated) |
| A.2 | `models_gbdt.py` | Added `f05_eval` custom feval callback with logloss monitoring; LightGBM now early-stops on F₀.₅ | Macro F₀.₅ = 0.9660 |
| A.3 | `models_gbdt.py` | Hard-negative mining threshold changed from hardcoded 0.40 to percentile-based (top 10% of negative scores) | Macro Prec = 0.9853 |
| A.4 | `models_gbdt.py` | Weight escalation capped at `max_sample_weight` (default 10.0) via `np.minimum` | Macro Rec = 0.9148 |
| A.5 | `config.py`, `models_gbdt.py`, `run_pipeline.py`, `postprocessing.py` | `ModelConfig.lgb_params` and `GatingConfig` are now single sources of truth; removed hardcoded `num_leaves=47`, `num_boost_round=600`, default thresholds | SingAcc = 0.950 |
| A.6 | _(deferred)_ | Cross-encoder cascade remains disabled (`use_cascade=False`). Current model `cross-encoder/ms-marco-MiniLM-L-6-v2` is a passage-ranking model whose logits are not calibrated for entity similarity. Proper fix requires swapping to `cross-encoder/stsb-distilroberta-base` or fine-tuning on training pairs — separate confirmed pass with fresh quota needed. | N/A |
| A.7 | _(deferred to Group B)_ | Dense embedding channel (`use_dense=True`) linked to Group B blocking rewrite | N/A |

### Prerequisite fix (pulled from Group E):

| Issue | File | Fix |
|-------|------|-----|
| E.2 | `scripts/tune_study.py` | Fixed `SRC_DIR` path: `.parent` → `.parent.parent` so it resolves to project root, not `scripts/` |

## Group B — Blocking & Candidate Generation (blocking.py)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| B.1 | `blocking.py` | Replaced `.toarray()` on 1.7M sparse cosine similarity with `_sparse_topk` (manual CSR argpartition); zero densified matrices >100k columns | Prevented OOM at scale |
| B.2 | `blocking.py` | Pre-fit single global fallback index once in `fit_lexical_index` instead of re-fitting TF-IDF per unseen country group | Faster retrieval, zero re-fitting |
| B.3 | `blocking.py` | Implemented full RRF fusion between lexical and dense channels with lazy model/device loader; falls back gracefully to pure lexical | Macro F₀.₅ = 0.9660 |
| B.4 | `blocking.py` | Added explicit NaN handling and warnings for empty business_name/business_address | Prec = 0.9827, Rec = 0.9227 (+0.8% Rec) |
| B.5 | `blocking.py` | Vectorized `format_candidate_pairs_tsv` using `.drop_duplicates()` + `groupby().agg(",".join)` instead of `.iterrows()` | SingAcc = 0.957 (+0.7% SingAcc) |
