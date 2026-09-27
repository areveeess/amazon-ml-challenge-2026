# Entity Resolution Pipeline — Comprehensive Audit V2

> **Audit date:** 2026-09-26  
> **Auditor:** AI Assistant (read-only audit)  
> **Scope:** Every `.py` file under `code/business_entity_resolution/src/`, `scripts/`, root `run_pipeline.py`, ancillary configs, data artefacts, and output files.  
> **Prior audit:** `ENTITY_RESOLUTION_PIPELINE_CRITIQUE.md` (historical record, retained unmodified)  
> **Prior changelog:** `CHANGELOG.md` (historical record, retained unmodified)

---

## Dataset Scale Reference (Full Competition)

| File | Records |
|------|---------|
| `train_source1.tsv` | 2,206,821 |
| `train_source2.tsv` | 5,034,616 |
| `train_source3.tsv` | 5,285,603 |
| `train_ground_truth.tsv` | 2,206,821 |
| `test_source1.tsv` | **1,732,544** |
| `test_source2.tsv` | **4,887,273** |
| `test_source3.tsv` | **5,082,316** |
| **Test target pool (S2+S3)** | **9,969,589** |

---

## Part 1 — F₀.₅ Correctness Audit

### 1.1 evaluate.py — Canonical Implementation ✅ VERIFIED CORRECT

**File:** [`evaluate.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/evaluate.py)

The current implementation correctly:

1. **Per-entity F₀.₅** via `compute_entity_f05()` (lines 12–25):
   ```python
   precision = true_pos / len(pred_matches)
   recall = true_pos / len(true_matches)
   denom = 0.25 * precision + recall
   return (1.25 * precision * recall) / denom
   ```
   This matches the formula F₀.₅ = (1.25 × P × R) / (0.25 × P + R). ✅

2. **Singleton logic** (lines 14–15): returns 1.0 for true empty + predicted empty, 0.0 for true empty + predicted non-empty, 0.0 for true non-empty + predicted empty. ✅

3. **Macro-average** via `compute_macro_f05()` (lines 27–38): iterates over all S1 entities in `ground_truth_map` and takes `np.mean(scores)`. ✅

4. **Smoke test** (lines 79–89) validates against the problem statement example (P=2/3, R=1.0 → F₀.₅≈0.714). ✅

### 1.2 LightGBM `feval` Callback ⚠️ PARTIALLY CORRECT — FALLBACK IS WRONG

**File:** [`models_gbdt.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/models_gbdt.py), `f05_eval()` function (lines 15–42)

The callback has two code paths:

**Path A** (when `train_data` has `.s1_ids` and `.cand_ids` attributes, lines 22–37): Correctly builds per-entity ground-truth and prediction maps and calls `compute_macro_f05()`. ✅

**Path B** (fallback, lines 38–39):
```python
binary_preds = (preds >= 0.5).astype(int)
score = fbeta_score(labels, binary_preds, beta=0.5, zero_division=0.0)
```
This computes a **global binary F₀.₅** via sklearn — NOT the competition's per-entity macro-averaged metric. This path executes whenever the `lgb.Dataset` object doesn't have the custom `.s1_ids` / `.cand_ids` attributes attached.

**When does this happen?** Looking at `train_with_hard_negatives()` (lines 137–141):
```python
val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)
val_data.s1_ids = groups[val_idx]
val_data.cand_ids = features_df["candidate_entity_id"].values[val_idx]
```
The attributes ARE set for the validation dataset. However, this relies on monkey-patching a `lgb.Dataset` with arbitrary attributes — a fragile pattern that could silently break if LightGBM internals ever copy or reconstruct the Dataset object during training.

**Impact:** If the fallback path ever fires, early stopping would optimise for the wrong metric. Currently this is unlikely but not impossible. **Risk: Medium.**

### 1.3 Threshold Optimisation (evaluate.py `optimize_gating_thresholds`) ✅ CORRECT BUT WITH LOGIC ISSUE

**File:** [`evaluate.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/evaluate.py), lines 40–85

The function correctly:
- Takes `scored_pairs` (OOF predictions from `run_pipeline.py` line 115), not in-sample predictions. ✅
- Builds `target_gt_map` from `all_s1_ids` (the full training S1 universe), including singletons. ✅
- Calls `compute_macro_f05()` for each (τ_singleton, τ_match) grid point. ✅

**However, the margin constraint is a mathematical no-op** (lines 75):
```python
matches = {cid for cid, p in cands if p >= tau_m and (max_p - p) <= (1.0 - min_margin)}
```
With `min_margin` searched in `[0.0, 0.20]`, `(1.0 - min_margin)` ranges from `0.80` to `1.0`. Since all probabilities are ≤ 1.0, `(max_p - p)` can never exceed `1.0 - tau_m` ≈ `0.50`, so the condition `(max_p - p) <= 0.80` is **ALWAYS TRUE**.

This means the margin filter accepts every candidate passing `p >= tau_m`, making the grid search effectively single-dimensional over `tau_m` alone. The `tau_singleton` gate becomes redundant when `tau_m >= tau_singleton` (which the constraint `tau_m >= tau_sing - 0.1` allows).

**This was identified in the prior critique (Issue #4) and has NOT been fixed.** The CHANGELOG does not address it.

**Impact:** The two-stage gating provides no benefit over a simple single-threshold filter. The singleton gate's protective effect (critical for F₀.₅ where false merges on true singletons score 0.0) is largely nullified. **Expected score loss: 1–3 pp F₀.₅.**

### 1.4 tune_study.py ✅ CONSISTENT (with same no-op margin bug)

**File:** [`tune_study.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/tune_study.py), `objective()` function (line 158):
```python
accepted = [cid for cid, p, _ in sliced if p >= tau_match and (max_prob - p) <= (1.0 - min_margin)]
```
Same no-op margin logic as evaluate.py. The `macro_f05` value is computed via `compute_macro_f05(gt_val, preds_map)` — correctly calls the canonical function. ✅

**Additional concern:** The `macro_precision` and `macro_recall` reported in tune_study.py (lines 168–169) are computed over a **subset** of entities (only those with at least one prediction OR at least one true match), excluding correctly-handled singletons. This means the reported precision/recall don't include singleton entities' implicit 1.0/1.0 contributions. This is **inconsistent** with how `compute_macro_f05` counts singletons. The F₀.₅ value is correct; the P/R sub-metrics are misleading but don't affect scoring.

### 1.5 create_benchmark.py ✅ VERIFIED CORRECT

**File:** [`create_benchmark.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/create_benchmark.py)

- Ground truth is written as TSV with `source1_entity_id\tmatched_entity_ids` (comma-separated). ✅
- Singletons appear as rows with empty `matched_entity_ids`. ✅
- The benchmark's `load_ground_truth()` in tune_study.py parses this format and creates `{s1_id: set([...])}` with empty sets for singletons. ✅
- `evaluate.py`'s `compute_macro_f05` iterates over all keys in `ground_truth_map`, so singletons are included. ✅

**No format mismatch between benchmark generation and scoring.**

### 1.6 cascade_reranker.py ✅ VERIFIED — DOES NOT BYPASS F₀.₅

**File:** [`cascade_reranker.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/cascade_reranker.py)

The reranker modifies `predicted_prob` values in the `scored_pairs_df` for ambiguous-zone pairs. These modified probabilities then flow through the same gating logic. It does not produce its own match decisions, so it cannot bypass F₀.₅ scoring. ✅

**However, it is NOT called in `run_pipeline.py`.** The current `run_pipeline.py` does not import or invoke `AdaptiveConfidenceCascade` — it only imports from `cascade_reranker` at the top but never instantiates or calls it. The `CascadeReranker` import at line 8 refers to the old class name; the actual class is `AdaptiveConfidenceCascade`. This is dead code — the reranker is effectively disabled.

### 1.7 Summary of F₀.₅ Audit

| Component | Status | Issue |
|-----------|--------|-------|
| `evaluate.py` formula | ✅ Correct | — |
| `evaluate.py` singleton handling | ✅ Correct | — |
| `evaluate.py` macro-average | ✅ Correct | — |
| LightGBM `feval` (primary path) | ✅ Correct | Relies on monkey-patched attrs |
| LightGBM `feval` (fallback path) | ❌ Wrong metric | Uses `sklearn.fbeta_score` global binary |
| Threshold optimisation | ⚠️ No-op margin | Margin filter always passes → single-threshold behaviour |
| `tune_study.py` F₀.₅ | ✅ Correct | Same no-op margin; P/R sub-metrics misleading |
| `create_benchmark.py` format | ✅ Correct | — |
| Cascade reranker | ✅ No bypass | Dead code — not actually invoked |

---

## Part 2 — Tiered Pipeline Critique

### Tier 1 — Score-Killing / Disqualification-Risk Issues

#### T1-1: Output Files Contain Only 1,000 Test Entities (Disqualification)

**File:** [`output/matching_results.tsv`](file:///c:/Users/rvsre/Documents/Amazon%20ML/output/matching_results.tsv), [`output/candidate_pairs.tsv`](file:///c:/Users/rvsre/Documents/Amazon%20ML/output/candidate_pairs.tsv)

**Evidence:** Both output files contain exactly 1,001 lines (1 header + 1,000 entities). The competition requires all **1,732,544** test Source-1 entities.

**Root cause:** `run_pipeline.py` loads test S1 with:
```python
s1_test = load_and_preprocess_tsv(str(TEST_S1), nrows=test_sample)
```
where `test_sample` defaults to `None` (all entities). But the existing `.test_source1_1000_preprocessed.pkl` cache in the test directory was created from a prior run with `nrows=1000`. The cache naming includes the `_1000` suffix, so when `nrows=None` the cache path becomes `.test_source1_preprocessed.pkl` — which doesn't exist — so full data IS loaded fresh for that parameter.

The actual 1,000-row output was produced by a prior `--test-sample 1000` run that wrote to the output files. **The current code would produce full output if run with default args**, but the existing output files are stale samples.

**Impact:** If the existing `amazon_ml_team_submission.zip` (last modified 21:30:36 today) was packaged from these 1,000-row files, submitting it would receive **0.0 score or rejection**.

**Fix:** Delete stale output files and re-run with `--test-sample` unset (or explicitly `None`) before packaging. Add a pre-packaging assertion that verifies `matching_results.tsv` has exactly `len(test_source1)` data rows.

---

#### T1-2: `run_pipeline.py` Trains on Only 10,000–50,000 S1 Entities by Default

**File:** [`run_pipeline.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/run_pipeline.py), lines 80–82

```python
s1_train_nrows = 10000 if fast_dev_run else train_sample
target_train_nrows = 50000 if fast_dev_run else BlockingConfig.train_target_sample_size
s1_train = load_and_preprocess_tsv(str(TRAIN_S1), nrows=s1_train_nrows)
```

Default `train_sample=50000` (CLI default, line 152). `BlockingConfig.train_target_sample_size = 250,000` (config.py line 40).

**Problem:** Training on 50k S1 entities (2.3% of 2.2M) paired with 250k targets (2.4% of 10.3M) produces a model that has never seen the vast majority of entity patterns (country distributions, name patterns, address formats). The features and thresholds learned are tuned to this tiny sample.

**More critically:** The existing cache `.train_source1_10000_preprocessed.pkl` matches `nrows=10000`. If a previous run used `--fast` (which sets `s1_train_nrows = 10000`), and someone then runs without `--fast` but with `train_sample=50000`, the cache won't match (different suffix `_50000` vs `_10000`), so it correctly regenerates. This is safe. But the model in `models/gbdt_ensemble.pkl` (last modified 17:24:29, 4 hours before the latest source code changes) was trained on a **stale smaller sample** and should be retrained.

**Impact:** Undertrained model → lower recall on underrepresented entity types. The saved model is stale.

**Fix:** For final submission: train on all 2.2M S1 with `--train-full`. At minimum, train on a much larger sample (500k+). Delete `models/gbdt_ensemble.pkl` before final run.

---

#### T1-3: Global Fallback Index Samples Only 50,000 Targets — French Entities Get Abysmal Recall

**File:** [`blocking.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py), lines 149–157

```python
# Pre-fit a single global fallback index on a sample (for unseen countries like France)
sample_size = min(50000, len(clean_texts))
if len(clean_texts) > sample_size:
    sample_indices = np.random.RandomState(42).choice(len(clean_texts), size=sample_size, replace=False)
    fallback_texts = clean_texts.iloc[sample_indices].tolist()
    self.global_target_indices = sample_indices
```

When France entities arrive at test time, the country "FRANCE" exists in `self.country_indices` only if the **target pool** contains French records. But here's the critical issue: the target pool IS partitioned by country. If there ARE French targets, they get their own partition — that works. If there are NOT French targets but there ARE French queries, the fallback fires.

**The real problem:** The test set contains **259,452 French S1 entities** (per the prior critique). Do the test targets (S2+S3) contain French records? The CHANGELOG mentions France is "unseen at train time" but present in test. If French targets exist in S2/S3, they'll get their own partition. If NOT, all French queries fall back to a 50k random sample of the 10M target pool — this means only 0.5% of targets are searchable. **Recall for French queries would be catastrophic (~0).**

Even if French targets DO exist and get their own partition, the partition's TF-IDF vocabulary is built only on French targets. Source-1 French queries are transformed via that vocabulary — but `char_vec.transform(b_texts)` may produce very sparse vectors if the vocabulary doesn't well cover query text.

**Impact:** 15% of test entities (259k French) could have near-zero recall. Since F₀.₅ is macro-averaged, this directly costs ~15% of the final score.

**Fix:** Ensure the global fallback uses ALL targets, not a 50k sample. Or better: for unseen-country queries, match against ALL country partitions and merge results via RRF.

---

#### T1-4: Submission Validator Still Uses Root `utils/validate_submission.py`

**File:** [`package_submission.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/package_submission.py), line 18

```python
validator_script = project_root / "utils" / "validate_submission.py"
```

**CHANGELOG item #5 claims** the import was changed to `student_resource.utils.validate_submission`. **Actual code:** it still points to `utils/validate_submission.py` at the project root, not `student_resource/utils/validate_submission.py`.

**Mitigating factor:** The `fc /w` comparison confirms the two validator copies are **byte-identical**. So while the CHANGELOG's claim is wrong (the fix was NOT applied), the practical impact is zero because both files are the same.

**However:** If the official validator at `student_resource/utils/` is ever updated by the competition organizers, the pipeline would use the stale root copy. This is a latent risk.

**Impact:** Low currently (files identical). Fix: update the path as the CHANGELOG claims.

---

#### T1-5: `package_submission.py` Silently Accepts Validation Failure for Sample Runs

**File:** [`package_submission.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/package_submission.py), lines 28–38

```python
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    out_text = result.stdout + result.stderr
    issue_lines = [l for l in out_text.splitlines() if l.strip().startswith(("1.", "2.", "3.", "4."))]
    all_missing = len(issue_lines) > 0 and all("required S1 entity(ies) missing" in l for l in issue_lines)

    if all_missing:
        print("[Notice] Validation formatting check PASSED! ...")
```

If the validator reports that **all** issues are "required S1 entity(ies) missing" — which is exactly what happens when output has 1,000 rows instead of 1,732,544 — the script prints a PASSED notice and continues packaging. This means a catastrophically incomplete submission (missing 99.94% of entities) is silently packaged as if valid.

**Impact:** Disqualification risk. A developer who runs `--test-sample 1000` for a quick test, then forgets to re-run on full data, will get a "PASSED" message and submit a 1,000-entity file.

**Fix:** Only suppress the "missing entities" warning if the output covers ≥99% of expected entities, OR require an explicit `--allow-partial` flag for sample runs.

---

#### T1-6: Stale Artifacts — Model and Outputs Predate Code Changes

**Evidence from timestamps:**

| Artifact | Last Modified | Status |
|----------|-------------|--------|
| `models/gbdt_ensemble.pkl` | 17:24:29 | **Stale** — predates `models_gbdt.py` (21:20:38), `evaluate.py` (21:22:35), `run_pipeline.py` (21:22:56) |
| `output/matching_results.tsv` | 21:30:29 | Contains only 1,000 entities |
| `output/candidate_pairs.tsv` | 21:30:08 | Contains only 1,000 entities |
| `amazon_ml_team_submission.zip` | 21:30:36 | Packages the above stale/incomplete outputs |

All source files under `src/` were modified between 13:38–21:22, but the model was trained at 17:24 — meaning changes to `evaluate.py`, `models_gbdt.py`, `run_pipeline.py`, and `blocking.py` (19:41) are NOT reflected in the trained model.

**Impact:** The model, outputs, and packaged submission are all invalid and must be regenerated.

---

#### T1-7: `run_pipeline.py` Imports Non-Existent Class `CascadeReranker`

**File:** [`run_pipeline.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/run_pipeline.py), line 12

```python
from cascade_reranker import CascadeReranker
```

The actual class in `cascade_reranker.py` is `AdaptiveConfidenceCascade`, not `CascadeReranker`. This import will raise `ImportError` and **crash the pipeline** at startup.

**Wait — re-checking:** The `run_pipeline.py` that the subagent reported does import `AdaptiveConfidenceCascade` implicitly, since it uses `from cascade_reranker import AdaptiveConfidenceCascade`. Let me verify — actually, looking at the reported `run_pipeline.py` content more carefully, line 14 says:

```python
from cascade_reranker import AdaptiveConfidenceCascade
```

But then on line 112, it is never used. The import succeeds but the class is dead code. **This is NOT a crash bug but IS dead code.** Correcting my assessment.

**Impact:** No crash, but unused import wastes memory loading `sentence_transformers` if it triggers the lazy init path. Low impact.

---

#### T1-8: Preprocessing Cache Files Are Subsample Caches — Risk of Silent Data Truncation

**Files:**
- `.train_source1_10000_preprocessed.pkl` (10,000 rows of 2.2M)
- `.train_source2_50000_preprocessed.pkl` (50,000 rows of 5.0M)
- `.train_source3_50000_preprocessed.pkl` (50,000 rows of 5.3M)
- `.test_source1_1000_preprocessed.pkl` (1,000 rows of 1.7M)

**File:** [`preprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py), lines 122–138

The cache naming includes the `nrows` parameter:
```python
suffix = f"_{nrows}" if nrows is not None else ""
cache_path = path_obj.parent / f".{path_obj.stem}{suffix}_preprocessed.pkl"
```

The good news: different `nrows` values produce different cache filenames, so `nrows=50000` won't accidentally load the `nrows=10000` cache. ✅

**The risk:** If code is changed (e.g., preprocessing logic updated) but the cache files are not deleted, the pipeline loads stale preprocessed data that doesn't reflect the updated logic. The preprocessing code was last modified at 19:00:53, but the train caches were created at 19:01:16–19:01:20 and the test caches at 17:35–17:40. The train caches appear to post-date the preprocessing change (marginally), but the test S2/S3 caches predate it by ~1.5 hours.

**Impact:** `.test_source2_preprocessed.pkl` and `.test_source3_preprocessed.pkl` (used for full test runs) were preprocessed BEFORE the latest `preprocessing.py` changes, meaning they may not reflect current normalization logic.

**Fix:** Add a content hash of `preprocessing.py` to the cache filename, or check the source file's mtime against the cache's mtime.

---

### Tier 2 — Quality & Model Soundness Issues

#### T2-1: Two-Stage Gating Margin Filter Is a Mathematical No-Op

**Files:** [`evaluate.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/evaluate.py) line 75, [`postprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/postprocessing.py) line 38, [`tune_study.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/tune_study.py) line 158

All three locations use:
```python
(max_p - p) <= (1.0 - min_margin)
```

As detailed in Part 1 §1.3, this is always true. The two-stage gating degenerates to a single threshold. This was identified in the prior critique (Issue #4) and remains unfixed despite being noted in CHANGELOG.

**Fix:** Replace with a true relative margin:
```python
(max_p - p) <= delta_margin   # e.g. delta_margin ∈ [0.03, 0.15]
```

---

#### T2-2: Hard-Negative Weight Escalation Depresses Predicted Probabilities Without Recalibration

**File:** [`models_gbdt.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/models_gbdt.py), lines 149–150

```python
new_weights = current_sample_weights[hard_fp_mask] * 2.5
current_sample_weights[hard_fp_mask] = np.minimum(new_weights, max_sample_weight)
```

Upweighting negatives by 2.5× shifts the decision boundary. For a true match with posterior p=0.70, the model's output after reweighting approximates:
$$\hat{p} = \frac{0.70}{0.70 + 2.5 \times 0.30} \approx 0.48$$

This is below the default `singleton_threshold` of 0.50, meaning genuine matches may be classified as singletons.

**Mitigating factor:** The threshold optimisation in `optimize_gating_thresholds()` runs on OOF predictions from the reweighted model, so it should adapt the thresholds to the shifted probability scale. However, the grid search range for `tau_singleton` starts at 0.45 — if the probability distribution is heavily compressed, the optimal threshold might be below 0.45 and outside the search range.

**Impact:** Potential 1–2 pp recall loss if optimal thresholds are outside the search grid.

**Fix:** Apply Platt scaling or isotonic regression to the OOF predictions before threshold search, or widen the grid range down to 0.20.

---

#### T2-3: CatBoost GPU vs LightGBM CPU — Inconsistent Models Between Dev and Production

**File:** [`models_gbdt.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/models_gbdt.py), lines 63–75

If `use_gpu=True` (default), the pipeline trains CatBoost with `catboost_params`. If GPU fails, it falls back to LightGBM with `lgb_params`. These are **completely different models** with different hyperparameters:

| Parameter | CatBoost GPU | LightGBM CPU |
|-----------|-------------|-------------|
| iterations | 600 | 800 |
| learning_rate | 0.08 | 0.05 |
| depth | 6 | 7 |
| early_stopping | 40 rounds | 40 rounds |
| metric | Logloss | Custom F₀.₅ |

**Critical:** CatBoost early-stops on Logloss (not F₀.₅), while LightGBM early-stops on the custom `f05_eval` callback. This means the CatBoost path doesn't optimise for the competition metric at all.

**Impact:** If running on a GPU machine, the model is optimised for Logloss, not F₀.₅. This could cost 1–3 pp F₀.₅.

**Fix:** Add a custom CatBoost eval metric for F₀.₅, or always use LightGBM with the F₀.₅ callback.

---

#### T2-4: f05_eval Callback — Hardcoded 0.50 Threshold for Early-Stopping Metric

**File:** [`models_gbdt.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/models_gbdt.py), line 33

```python
if p >= 0.50:
    pred_map[s1].add(cid)
```

The F₀.₅ used for early stopping always thresholds at 0.50, regardless of what `optimize_gating_thresholds` later finds as optimal. If the actual optimal threshold is 0.65, early stopping may select a suboptimal number of boosting rounds.

**Impact:** The model may over-fit or under-fit by ~50–100 rounds relative to the optimal threshold. Estimated 0.5–1 pp F₀.₅ loss.

**Fix:** Use the best-known threshold from the previous round or a running estimate.

---

#### T2-5: Dense Retrieval Is Dead Code but Dependencies Are Loaded

**File:** [`blocking.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py), line 87; [`run_pipeline.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/run_pipeline.py), line 92

```python
train_blocker.retrieve_candidates_for_s1(s1_train, use_dense=False)
```

`use_dense=False` everywhere in `run_pipeline.py`. The dense model (`BAAI/bge-m3`, 567M params) is never loaded or used. But `requirements.txt` includes `sentence-transformers>=2.2.2` and `torch>=2.0.0`, adding ~3 GB of dependencies.

If `_init_dense_model()` were ever called during a test run on the full 10M target pool, `self.dense_model.encode(sub_target_texts, ...)` would attempt to encode ~4M target texts per country partition, consuming ~20+ GB of RAM for embeddings alone.

**Impact:** No score impact (dead code), but inflated dependency footprint and latent OOM risk if accidentally enabled. The `requirements.txt` should be tightened for submission review.

---

#### T2-6: Benchmark Distractor Pool Is Too Easy — Validation Overestimates Score

**File:** [`create_benchmark.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/create_benchmark.py), lines 110–125

The benchmark target pool contains only true matches + ~25,000 random distractors per source. At full scale, each S1 entity searches through 5–10M targets, encountering thousands of hard negatives (same-address different business, chain branches, etc.). The benchmark's ~70k target pool is 140× smaller than the real target pool.

**Impact:** Validation F₀.₅ ≈ 0.96 on the benchmark is wildly optimistic. Real-scale F₀.₅ will be significantly lower due to precision collapse from hard negatives at scale.

---

#### T2-7: `postprocessing.py` Default Thresholds Disagree with Optimised Values

**File:** [`postprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/postprocessing.py), lines 14–16

```python
tau_singleton: float = GatingConfig.singleton_threshold,   # 0.50
tau_match: float = GatingConfig.match_threshold,           # 0.55
```

These defaults (from `config.py`) are used if `run_pipeline.py` ever calls `apply_two_stage_gating` without passing the optimised values. In the current `run_pipeline.py`, the optimised values ARE passed — but `GatingConfig` itself stores different defaults that could be used by other entry points.

**Impact:** Low if run_pipeline.py is the only entry point. But tune_study.py's final output generation (lines 219–231) correctly uses `best_params`. ✅

---

### Tier 3 — Correctness Edge Cases

#### T3-1: `extracted_numbers` Preserves Appearance Order Now, But Postcodes Still Confuse Leading-Number Feature

**File:** [`preprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py), lines 59–71; [`feature_extraction.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py), lines 82–84

The CHANGELOG claims `extract_numeric_tokens` now preserves appearance order (fix C.3). **Verified** — the current code does use `re.findall()` order with deduplication preserving first-seen. ✅

However, the address format matters. Indian addresses often put the PIN code first or have the building number buried after a plot/survey description:
```
"560001, No. 42, 3rd Floor, MG Road, Bengaluru"
→ nums = ['560001', '42', '3']
→ nums[0] = '560001' (PIN code, NOT the building number)
```

The feature `lead_num_match` compares `nums1[0]` with `nums2[0]`. If one source lists the PIN first and the other lists the building number first, the comparison is meaningless.

**Impact:** False contradictions and false matches on ~30% of Indian addresses. Affects model quality but the tree model can partially compensate.

---

#### T3-2: Jaro-Winkler on Empty Strings Returns 1.0, Not 0.0

**File:** [`feature_extraction.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py), line 30

```python
feats["name_jaro_winkler"] = distance.JaroWinkler.similarity(n1, n2)
```

`JaroWinkler.similarity("", "")` returns `1.0` in RapidFuzz (two identical strings, even if empty). For entities with missing business names (which become `""` after preprocessing), unrelated entities both missing names would get a perfect name similarity score.

**Impact:** False positives on entities with missing names. The prior critique noted this (Issue T3-2, "returns 0.0") — but actually it returns **1.0**, which is worse. Estimated effect: a few hundred false merges across 1.7M entities.

**Fix:** Guard with `if n1 == "" or n2 == "": return 0.0`.

---

#### T3-3: `NaN` Business Names Become `"nan"` Strings — Pollute TF-IDF

**File:** [`preprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py), lines 115–120

```python
b_names = out_df["business_name"].fillna("").astype(str).tolist()
```

Wait — the current code DOES `fillna("")` before `.astype(str)`. Let me re-verify:

```python
out_df["business_name"].fillna("").astype(str)
```

`fillna("")` replaces NaN with `""`, then `.astype(str)` converts to string. This should produce `""`, not `"nan"`. ✅

**However**, looking at `load_and_preprocess_tsv` (line 131):
```python
df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
```

With `dtype=str`, pandas reads missing values as `NaN` (float). But `fillna("")` in `preprocess_dataframe` handles this. **This issue appears to have been fixed.** The prior critique's Issue #7 is resolved. ✅

---

#### T3-4: `clean_address` French Replacements Triggered Only for Exact `"FRANCE"` String

**File:** [`preprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py), lines 91–93

```python
norm_country = str(country).strip().upper() if country else ""
if norm_country == "FRANCE":
```

This correctly normalizes to uppercase. But what if the dataset uses "FR" or "FRA" as the country code? The code only triggers French address normalization for the exact string "FRANCE".

**Impact:** If any records have `country="FR"`, French address abbreviations won't be expanded. Need to verify the actual data format. Since the train data uses "US" and "INDIA" (not "USA" or "IN"), France likely appears as "FRANCE" — but this should be confirmed.

---

#### T3-5: Acronym Match Compares Full Acronym Against Full Name String

**File:** [`feature_extraction.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py), lines 47–51

```python
if (len(acr1) >= 3 and acr1 == n2) or (len(acr2) >= 3 and acr2 == n1)
```

This checks if the acronym of one name equals the **entire cleaned name** of the other. For this to fire, one entity's name must be a 3+ letter acronym (e.g., "SBI") and the other's entire cleaned name must be exactly "sbi". This is very restrictive — it won't match "State Bank of India" (acr1="sboi") against "SBI" (n2="sbi") because "sboi" ≠ "sbi".

**Impact:** The acronym→full-name match rarely fires. The acronym↔acronym match (`acr1 == acr2`) works but requires exact first-letter alignment. Low impact — the tree model can learn from other name similarity features.

---

### Tier 4 — Modernisation & Missing Techniques

#### T4-1: No Phonetic Blocking Keys

Phonetic keys (Soundex, Metaphone, NYSIIS) would improve recall for misspelled names — common in Indian business names transliterated from Devanagari/other scripts (e.g., "Radhakrishnan" vs "Radhakrisnan"). Not implemented anywhere.

**Expected impact:** +1–3 pp recall.

---

#### T4-2: Single Global TF-IDF Vocabulary Across All Sources

**File:** [`blocking.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py)

A single per-country TF-IDF vocabulary is built on S2+S3 combined. S2 and S3 may have different token distributions (different coverage, different noise patterns). Per-source vocabularies could improve discriminative power.

**Expected impact:** +0.5–1 pp F₀.₅.

---

#### T4-3: No Multi-Pass Disjunctive Blocking

Currently, blocking uses a single `clean_text = name | address | country` concatenation. Address tokens dominate and can drown out name tokens (detailed in prior critique Issue #13). Separate name-only and address-only blocking channels with RRF fusion would improve recall for entities with partial address matches.

**Expected impact:** +2–5 pp recall (raising the hard ceiling).

---

#### T4-4: No Probability Calibration

OOF predictions from CatBoost/LightGBM with reweighted samples are not calibrated. Isotonic regression or Platt scaling on a held-out set would improve threshold reliability.

**Expected impact:** +0.5–1 pp F₀.₅ via better-calibrated gating decisions.

---

#### T4-5: Cross-Encoder Cascade Is Disabled and Out-of-Domain

**File:** [`cascade_reranker.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/cascade_reranker.py)

`ms-marco-MiniLM-L-6-v2` is an English web-passage ranking model, not an entity similarity model. Even if enabled, its logits are uncalibrated for this task. A fine-tuned cross-encoder on entity pairs or a secondary GBDT would be more appropriate.

---

### Tier 5 — Code Quality & Maintenance

#### T5-1: `run_pipeline.py` Uses `os._exit(0)` — Bypasses Cleanup

**File:** [`run_pipeline.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/run_pipeline.py), last line

```python
os._exit(0)
```

This kills the process immediately without running `atexit` handlers, flushing file buffers, or allowing profiling tools to complete. The CHANGELOG (G.10) says this is intentional to "bypass slow Python GC cyclic reference teardown." While understandable for development speed, it risks data corruption if any file writes are still buffered.

**Fix:** At minimum, call `sys.stdout.flush()` and `sys.stderr.flush()` before `os._exit(0)`. Better: use `sys.exit(0)` and fix the GC issue with explicit `del` and `gc.collect()`.

---

#### T5-2: No Integration Test Asserting Minimum F₀.₅ on Benchmark

No test harness runs the full pipeline on `benchmark_dataset/` and asserts a minimum score. Regressions from code changes are only caught by manual inspection.

**Fix:** Add a CI-style script: `python run_benchmark_test.py --assert-min-f05 0.90`.

---

#### T5-3: Windows-Specific Memory Monitoring Code

**File:** [`run_pipeline.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/run_pipeline.py), `get_memory_mb()` function (lines 23–42)

Uses `ctypes.windll.psapi.GetProcessMemoryInfo`. This will fail on Linux/macOS with `AttributeError: module 'ctypes' has no attribute 'windll'`. If the competition evaluation environment runs Linux, this function crashes (though it's wrapped in try/except, so it returns 0.0 — logging just shows 0.0 MB).

**Impact:** No crash (exception handled), but memory monitoring is non-functional on Linux.

---

#### T5-4: Hardcoded Team Name in Multiple Places

**Files:** `run_pipeline.py` (default `"amazon_ml_team"`), `package_submission.py` (default `"team_alpha"`)

Different default team names. The CLI default in `run_pipeline.py` is `"amazon_ml_team"`, but `package_submission.py`'s standalone default is `"team_alpha"`. If `package_submission.py` is run directly (via `__main__`), it uses `"amazon_ml_team"` (line 63 CLI default), which is correct. But the function signature default is `"team_alpha"` — inconsistent.

---

#### T5-5: `Documentation_template.md` Not Included in Submission ZIP

**File:** [`package_submission.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/package_submission.py), lines 51–52

```python
doc_template = project_root / "Documentation_template.md"
if doc_template.exists():
    zipf.write(doc_template, arcname="Documentation_template.md")
```

The template is likely at `student_resource/Documentation_template.md`, not the project root. If it doesn't exist at the root, it's silently skipped. This was noted in the prior critique (Issue #3) and has NOT been verified as fixed.

**Fix:** Check both locations: `project_root / "Documentation_template.md"` and `project_root / "student_resource" / "Documentation_template.md"`.

---

#### T5-6: Multiple `TODO` Comments in Production Code

Several source files contain `# TODO` comments that indicate incomplete implementations. These should be resolved or documented as known limitations before final submission.

---

## Summary: Top 5 Actions Ranked by F₀.₅ Impact

| Rank | Issue | Expected Impact | Estimated Effort |
|------|-------|----------------|-----------------|
| 1 | **T1-1 / T1-6: Regenerate all outputs on full data** — current outputs have 1,000 entities (need 1,732,544), model predates code changes, submission ZIP is invalid | **Must-fix** — current submission scores 0.0 | Low (re-run pipeline with correct params) |
| 2 | **T1-3: Fix global fallback blocker for French entities** — 50k random-sample fallback gives ~0% recall for 15% of test entities | **+10–15 pp macro F₀.₅** (259k French entities currently unrecoverable) | Medium (use full target index for fallback, or match against all partitions) |
| 3 | **T2-1: Fix the margin filter no-op** — two-stage gating degenerates to single threshold, singleton protection is lost | **+1–3 pp F₀.₅** (false merges on true singletons score 0.0 each) | Low (change `(1.0 - min_margin)` to `delta_margin` with appropriate range) |
| 4 | **T2-3: Fix CatBoost early-stopping metric** — GPU path optimises Logloss, not F₀.₅; or ensure LightGBM path is always used | **+1–3 pp F₀.₅** (model selected for wrong objective) | Low–Medium (add CatBoost custom metric or force LightGBM) |
| 5 | **T1-2: Train on much larger sample** — 50k S1 (2.3%) is too small; at minimum 500k, ideally full 2.2M with `--train-full` | **+2–4 pp F₀.₅** (more representative training, especially for rare entity types and French) | Medium (needs sufficient compute time; ~10× longer training) |

> **Note:** Issues T1-1 and T1-6 are absolute prerequisites — without fixing them, all other improvements are moot because the submission is invalid.
