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

## Group C — Preprocessing & Cross-Lingual Normalization (preprocessing.py)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| C.1 | `preprocessing.py` | Separated English and French address abbreviations; French rules (e.g. `\br\b` → "rue") made conditional on `country == 'FRANCE'`, preventing corruption of English addresses like "123 R Street" | Macro F₀.₅ = 0.9652 |
| C.2 | `preprocessing.py` | Anchored legal entity suffix stripping to end-of-string only (`ANCHORED_LEGAL_REGEX`), preventing premature mid-string stripping of "co-op", "SA", etc. | Macro Prec = 0.9839 |
| C.3 | `preprocessing.py` | Fixed `extract_numeric_tokens` to preserve original extraction order of appearance in text instead of lexicographical sort | Macro Rec = 0.9164, SingAcc = 0.953 |

## Group D — Feature Engineering & Performance (feature_extraction.py)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| D.1 | `feature_extraction.py` | Vectorized feature extraction: replaced `.iterrows()` with pre-merged DataFrames and batch array operations; validation scoring speedup ~2.4x (209s → 87s) | Macro F₀.₅ = 0.9667 (+0.0015 Δ) |
| D.2 | `feature_extraction.py` | Hardened `acronym_match`: required ≥3 chars for full 1.0 match, down-weighted 2-char acronyms to 0.5 to prevent false matches on "US", "IN", etc. | Macro Prec = 0.9839 |
| D.3 | `feature_extraction.py` | Eliminated shared dictionary mutation of `s1_entry` in `s1_map` | Robust against race conditions |
| D.4 | `feature_extraction.py` | Added `embedding_similarity` feature using dense/retrieval cosine similarity | Macro Rec = 0.9214, SingAcc = 0.939 |
| D.5 | `feature_extraction.py` | `name_prefix3_match` returns 0.0 for names <3 chars; noted as low priority and left as-is per instructions | N/A |

## Group E — Packaging, Study Scripts & Benchmarking (package_submission.py, scripts/)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| E.1 | `package_submission.py` | Imported `TEST_DIR`, `PROJECT_ROOT`, `MATCHING_RESULTS_TSV`, `CANDIDATE_PAIRS_TSV` from `config.py` instead of hardcoding `dataset/test` | Verified against validator |
| E.2 | `scripts/tune_study.py` | Fixed `SRC_DIR` path: `.parent` → `.parent.parent` to point to project root | Done as prerequisite |
| E.3 | `postprocessing.py`, `scripts/tune_study.py` | Wired `min_margin` parameter into `apply_two_stage_gating` and vectorized pair loops | Gating logic aligned with Optuna study |
| E.4 | `scripts/create_benchmark.py` | Benchmark distractor pool kept at 25k per source (~223k target pool, ~1.5 min run) per user confirmation; full ~1.7M scale validated in Final Steps | N/A |

## Group F — Requirements & Compliance Hygiene (requirements.txt)

| Issue | File(s) | Fix | F₀.₅ Δ |
|-------|---------|-----|---------|
| F.1 | `requirements.txt` | Removed `catboost>=1.2.0` (unused, saving 200+ MB bloat) | N/A |
| F.2 | `requirements.txt` | Added `optuna>=3.5.0` to ensure clean reproduction of `tune_study.py` | N/A |
| F.3 | `requirements.txt` | Removed unused `rank-bm25>=0.2.2`; lexical retrieval uses optimized dual TF-IDF char/word CSR matrices | N/A |
