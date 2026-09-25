# Critical Technical Audit & Review: Business Entity Resolution Pipeline
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Evaluation Metric:** Macro $F_{0.5}$ (Precision-Weighted, $\beta = 0.5$) with Strict Singleton Penalty  
**Scope:** Full repository review (`src/`, `scripts/`, `output/`, configurations, dependencies, compliance)

---

## Executive Summary & Scalability Reality Check

The current entity resolution pipeline has significant architectural, mathematical, and algorithmic flaws that will prevent it from executing successfully or scoring well on the competition test set.

### The 10-Million Target Reality
Testing has previously been conducted only on an artificial ~70,000-record benchmark (`benchmark_dataset/`) containing only 25,000 random distractors. On the actual competition dataset:
- **`test_source1.tsv`**: **1,732,544** records (queries)
- **`test_source2.tsv`**: **4,887,273** records
- **`test_source3.tsv`**: **5,082,316** records
- **Combined Target Pool ($S2 + S3$)**: **9,969,589 records (~10 MILLION TARGETS)**
- **Country Distribution ($S1$ Test)**: India: 809,986 | US: 663,106 | France: 259,452

If run on the test dataset in its current state, the pipeline will **crash immediately with an Out-Of-Memory (OOM) error** when converting sparse matrix dot products to dense arrays (`.toarray()`), take **over 40 hours** due to multiple unvectorized pandas iteration anti-patterns, or be **disqualified / scored 0.0** due to formatting and entrypoint errors.

---

## Priority 0: Fatal Disqualification & Format Submission Killers

### 1. Missing Entrypoint: `run_pipeline.py` Does Not Exist
* **Location:** [`code/business_entity_resolution/README.md#L32`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/README.md#L32)
* **What’s wrong:** The documentation instructs evaluators and users to execute:
  ```bash
  python run_pipeline.py --team-name <your_team_name>
  ```
  However, **no file named `run_pipeline.py` exists** anywhere in `code/business_entity_resolution/` or in the project root.
* **Why it matters:** The official problem statement rules specify:
  > *"All teams submit a single zip archive with your code and outputs... Top teams' packages are reviewed in detail before the final rankings are confirmed... Anyone should be able to regenerate both output files from the training/test data using only what is in this folder."*
  An auditor attempting to reproduce your submission will hit `FileNotFoundError` immediately.
* **Remediation:** Create a self-contained driver script `code/business_entity_resolution/run_pipeline.py` that executes end-to-end inference on `dataset/test`, writes both `output/matching_results.tsv` and `output/candidate_pairs.tsv`, verifies them via `validate_submission.py`, and archives the submission package.

---

### 2. Output Files Contain Validation IDs, Not Test IDs
* **Location:** [`output/matching_results.tsv`](file:///c:/Users/rvsre/Documents/Amazon%20ML/output/matching_results.tsv) and [`output/candidate_pairs.tsv`](file:///c:/Users/rvsre/Documents/Amazon%20ML/output/candidate_pairs.tsv)
* **What’s wrong:** Running the official validator `student_resource/utils/validate_submission.py` against `student_resource/dataset/test` fails with 4 fatal errors:
  ```text
  FAIL — 4 issue(s) to fix before submitting:
    1. matching_results.tsv: required S1 entity(ies) missing: 1732544 total
    2. matching_results.tsv: row(s) using an S1 ID that is not in the test set: 10000 total
    3. candidate_pairs.tsv: required S1 entity(ies) missing: 1732544 total
    4. candidate_pairs.tsv: row(s) using an S1 ID that is not in the test set: 10000 total
  ```
* **Why it matters:** `tune_study.py` wrote its validation split predictions (10,000 entities from `benchmark_dataset/val`) directly into `output/`. Submitting this file to the portal results in an immediate **0.0 score or rejection**.
* **Remediation:** Separate internal experiment logs from competition outputs. Ensure validation predictions write to `output/val/`, while `output/matching_results.tsv` is reserved strictly for all 1,732,544 test $S1$ entities.

---

### 3. Packaging Script Silently Skips Validation and Drops Documentation
* **Location:** [`code/business_entity_resolution/src/package_submission.py#L28-L68`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/package_submission.py#L28-L68)
* **What’s wrong:**
  1. `test_dir` is hardcoded to `project_root / "dataset" / "test"`. Because the dataset is located at `student_resource/dataset/test`, `(test_dir / "test_source1.tsv").exists()` evaluates to `False`. The script prints:
     `[Notice] Test files not found in dataset/test yet. Skipping active validation check.` and silently bypasses formatting checks.
  2. `doc_template` looks for `project_root / "Documentation_template.md"`. The template is located at `student_resource/Documentation_template.md`. Line 66 evaluates to `False`, so **the zip file is packaged without the mandatory documentation file.**
* **Why it matters:** Incomplete zip submissions that lack methodology documentation or fail format rules are subject to disqualification.
* **Remediation:** Update `package_submission.py` to use dynamic path resolution (checking both `student_resource/` and root), fail with an explicit error code if validation fails or if `Documentation_template.md` is absent, and verify zip contents prior to completion.

---

## Priority 1: Correctness Bugs & Metric-Killing Logic Errors

### 4. Mathematical No-Op in the Confidence Margin Filter
* **Location:** [`scripts/tune_study.py#L158`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/tune_study.py#L158) & [`L250`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/tune_study.py#L250)
* **What’s wrong:** The condition to accept secondary candidates is:
  ```python
  accepted = [cid for cid, p, _ in sliced if p >= tau_match and (max_prob - p) <= (1.0 - min_margin)]
  ```
  `min_margin` was tuned between `0.0` and `0.20`. Consequently, `(1.0 - min_margin)` evaluates to between `0.80` and `1.0`.
  Because `max_prob <= 1.0` and `p >= tau_match` (e.g. $\ge 0.50$), `max_prob - p` can **never exceed 0.50**.
  Therefore, `(max_prob - p) <= 0.80` is **ALWAYS TRUE for every candidate passing `p >= tau_match`**.
* **Why it matters:** The margin filter does not filter any candidates. Furthermore, in [`src/postprocessing.py#L14`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/postprocessing.py#L14), `min_margin` isn't even exposed or implemented. Optuna spent 20 trials optimizing a parameter that had zero effect.
* **Remediation:** Implement true relative margin filtering:
  ```python
  # Keep secondary candidates only if they are within delta of the top candidate
  accepted = [cid for cid, p, _ in sliced if p >= tau_match and (max_prob - p) <= delta_margin]
  # where delta_margin is small (e.g. 0.05 to 0.15)
  ```

---

### 5. Contradictory Thresholds in `postprocessing.py` Invalidate Two-Stage Gating
* **Location:** [`code/business_entity_resolution/src/postprocessing.py#L17-L58`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/postprocessing.py#L17-L58)
* **What’s wrong:** The default arguments are `tau_singleton = 0.65`, `tau_match = 0.70`.
  Consider the logic:
  ```python
  if max_p < tau_singleton:
      predictions[s1_id] = []
  else:
      matched = [cid for cid, p in sorted(cands, ...) if p >= tau_match]
      predictions[s1_id] = dedup_matched
  ```
  If `max_p = 0.68`:
  1. `max_p < 0.65` is `False`.
  2. The `else` branch executes: it filters candidates where `p >= 0.70`.
  3. Since `max_p` is 0.68, no candidate has `p >= 0.70`. `matched` is `[]`.
  Any entity with `max_p < tau_match` produces `[]` regardless of `tau_singleton`. When `tau_match >= tau_singleton`, **the singleton gate is a redundant duplicate of the match gate.**
* **Why it matters:** Two-stage gating is designed to solve the singleton problem under $F_{0.5}$ (where false merges score 0.0). Its purpose is: require a high barrier to declare that an entity has *any* match ($\tau_{\text{singleton}} \approx 0.75$), but once confident it is not a singleton, allow close secondary matches ($\tau_{\text{match}} \approx 0.60$). Having $\tau_{\text{match}} > \tau_{\text{singleton}}$ reverses the intended mechanics.
* **Remediation:** Invert the hierarchy:
  ```python
  # Stage 1: Singleton Barrier (High precision check)
  if max_p < tau_singleton:  # e.g. 0.72
      predictions[s1_id] = []
  else:
      # Stage 2: Match Inclusion (Multi-source matches)
      predictions[s1_id] = [cid for cid, p in cands if p >= tau_match and (max_p - p) <= max_gap]
  ```

---

### 6. String-Sorted Numbers Invalidate "Leading Building Number" Alignment
* **Location:** [`code/business_entity_resolution/src/preprocessing.py#L96`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py#L96) & [`code/business_entity_resolution/src/feature_extraction.py#L85`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py#L85)
* **What’s wrong:** In `preprocessing.py`:
  ```python
  def extract_numeric_tokens(address: str) -> List[str]:
      numbers = re.findall(r"\b\d{1,6}\b", address)
      return sorted(list(set(numbers)))  # Lexicographical string sort
  ```
  In `feature_extraction.py`:
  ```python
  # Address Number Sequence Alignment: Leading building number match / contradiction
  if nums1 and nums2:
      feats["lead_num_match"] = 1.0 if nums1[0] == nums2[0] else 0.0
  ```
  Because `numbers` is sorted as strings:
  - Address 1: `"99 Broadway, Suite 1"` $\rightarrow$ `['1', '99']`. `nums1[0]` is `'1'` (suite number).
  - Address 2: `"1 Broadway, Suite 99"` $\rightarrow$ `['1', '99']`. `nums2[0]` is `'1'` (building number).
  - The feature evaluates to `lead_num_match = 1.0`.
  - Conversely, Address A: `"200 Main St"` $\rightarrow$ `['200']`. Address B: `"200 Main St, Apt 1"` $\rightarrow$ `['1', '200']`.
  - `nums1[0]` is `'200'`, `nums2[0]` is `'1'`. The feature evaluates to `lead_num_contradict = 1.0`.
* **Why it matters:** Building number matching/contradiction is a critical signal for distinguishing branches of chains (e.g. Starbucks at 100 Main vs 200 Main). Mismatching building numbers with suite numbers injects false positives and false negatives directly into the tree splits.
* **Remediation:** Preserve original token order:
  ```python
  def extract_address_numbers(address: str):
      numbers = re.findall(r"\b\d{1,6}\b", address)
      lead_num = numbers[0] if numbers else ""
      all_nums = set(numbers)
      return lead_num, all_nums
  ```

---

### 7. Missing Fields (`NaN`) Turn into Literal `"nan"` Strings
* **Location:** [`code/business_entity_resolution/src/preprocessing.py#L131-L142`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py#L131-L142)
* **What’s wrong:**
  ```python
  name_tuples = [clean_business_name(str(name)) for name in out_df["business_name"]]
  out_df["clean_address"] = [clean_address(str(addr)) for addr in out_df["business_address"]]
  out_df["clean_text"] = out_df["clean_name"] + " | " + out_df["clean_address"] + " | " + out_df["country"]
  ```
  `pd.read_csv(..., dtype=str)` sets missing fields to `float('nan')`. Calling `str(np.nan)` creates the string `"nan"`.
  `clean_business_name("nan")` returns `("nan", "")`.
  `clean_address("nan")` returns `"nan"`.
  `clean_text` becomes `"nan | nan | US"`.
* **Why it matters:** Any entity with a missing address or name will match other unrelated entities with missing fields based on the token `"nan"`. In TF-IDF, if `"nan"` appears frequently, it either pollutes candidate sets or inflates cosine similarity between completely disjoint businesses.
* **Remediation:** Clean missing values explicitly before string normalization:
  ```python
  out_df["business_name"] = out_df["business_name"].fillna("")
  out_df["business_address"] = out_df["business_address"].fillna("")
  ```

---

### 8. French Preprocessing Rules Corrupt US and Indian Addresses
* **Location:** [`code/business_entity_resolution/src/preprocessing.py#L42-L50`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py#L42-L50)
* **What’s wrong:** French abbreviations are in the global `ADDRESS_REPLACEMENTS` dict applied indiscriminately to all countries:
  ```python
  r"\ball\b": "allee",    # English: "All Saints Rd" -> "allee saints road"
  r"\br\b": "rue",        # English: "RR 1" / "Bldg R" -> "rue"
  r"\bchem\b": "chemin",  # English: "Chem Plant Rd" -> "chemin plant rd"
  r"\bpl\b": "place"      # English: "PL Tower" / "Pl."
  ```
  Furthermore, `LEGAL_PATTERNS` strips `\bsa\b` (Société Anonyme) globally. In Indian names, "SA Enterprises" or "SA Associates" has "sa" stripped, leaving "enterprises" (which is also stripped), reducing the name to an empty string.
* **Why it matters:** Mangling 809k Indian and 663k US addresses to cater to 259k French entities causes unintended feature drift on the training distribution.
* **Remediation:** Condition replacements on the country attribute:
  ```python
  if country == "FRANCE":
      # apply French regex replacements
  elif country in ("US", "INDIA"):
      # apply English / Indian regex replacements
  ```

---

### 9. Dense Retrieval is a Non-Functional Placebo
* **Location:** [`code/business_entity_resolution/src/blocking.py#L26-L35`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L26-L35) & [`L136`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L136)
* **What’s wrong:**
  1. `_init_dense_model()` imports `SentenceTransformer("BAAI/bge-m3")`.
  2. Inside `retrieve_candidates_for_s1()`, `self.dense_model.encode()` is **never called**.
  3. No dense index (FAISS, HNSW, or matrix) is ever built.
  4. The candidate rows hardcode:
     ```python
     "dense_rank": 1000
     ```
  5. The RRF calculation in line 134 uses only lexical rank:
     ```python
     rrf = 1.0 / (self.rrf_k + rank + 1)
     ```
* **Why it matters:**
  - The pipeline documentation claims a "Dual-Channel Dense Semantic Blocker," but it is running purely on character 3-gram and word 1-2 gram TF-IDF.
  - `requirements.txt` forces the installation of `torch`, `transformers`, and `sentence-transformers` (adding ~3GB of dependencies) for dead code.
* **Remediation:** Either implement true dense retrieval properly using an efficient sparse-dense index, or drop the dependency entirely to make the pipeline fast, lightweight, and deterministic.

---

### 10. Out-of-Domain English Cross-Encoder Blended with GBDT
* **Location:** [`code/business_entity_resolution/src/cascade_reranker.py#L26-L78`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/cascade_reranker.py#L26-L78)
* **What’s wrong:**
  1. The reranker loads `cross-encoder/ms-marco-MiniLM-L-6-v2`. This model was trained on **English web search passage retrieval** (MS-MARCO). It has zero domain knowledge of business entities, tax suffixes, legal structures, or Indian/French addresses.
  2. In line 74: `ce_probs = 1.0 / (1.0 + np.exp(-ce_scores))`. MS-MARCO logits are unbounded relevance scores (typically spanning $-10$ to $+10$), not log-odds of entity identity. Passing them through a sigmoid produces arbitrary, uncalibrated numbers.
  3. In line 77: `blended = 0.5 * gbdt_prob + 0.5 * ce_probs`. Linearly blending a calibrated tree model's probability with an uncalibrated passage retrieval score actively degrades the probability ranking in the ambiguous band.
  4. `ms-marco-MiniLM` is monolingual English. On the 259,452 French test entities, its WordPiece tokenizer fragments French words into character-level fragments.
* **Why it matters:** Ambiguous candidate pairs around the threshold are the most critical decisions for $F_{0.5}$. Blending in an uncalibrated English passage ranker will degrade precision.
* **Remediation:** Remove `cross-encoder/ms-marco-MiniLM-L-6-v2`. If you need a second-stage reranker, train a secondary gradient-boosted decision tree or ranker (or an `xlm-roberta` cross-encoder fine-tuned directly on entity resolution pairs).

---

## Priority 2: Catastrophic Performance & Scalability Bottlenecks

### 11. OOM Explosion on Dense Matrix Conversion (`.toarray()`)
* **Location:** [`code/business_entity_resolution/src/blocking.py#L114-L116`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L114-L116)
* **What’s wrong:** Look at the inner retrieval batch loop:
  ```python
  sim_char = (b_char @ char_mat.T).toarray()
  sim_word = (b_word @ word_mat.T).toarray()
  sim_lex = 0.5 * sim_char + 0.5 * sim_word
  ```
  Consider the US country partition in the test set:
  - Queries ($S1$ US): ~663,000 entities. Batch size = 500.
  - Targets ($S2 + S3$ US): ~3,900,000 entities.
  - `b_char @ char_mat.T` is a sparse matrix of dimension $(500, 3900000)$.
  - Calling `.toarray()` materializes a dense float64 array of shape $(500, 3900000)$:
    $$500 \times 3,900,000 \times 8 \text{ bytes} \approx 15.6 \text{ GIGABYTES PER BATCH}$$
  - `sim_word` allocates another 15.6 GB.
  - `sim_lex` allocates another 15.6 GB.
* **Why it matters:** On any standard machine (16GB or 32GB RAM), **the process will instantly crash with `MemoryError` on the very first batch.**
* **Remediation:** Never convert sparse retrieval dot products to dense arrays. Use C-accelerated sparse top-k pruning (e.g. `scipy.sparse` matrix-vector multiply with heap argpartition, or the standard `sparse_dot_topn` library), keeping only the top-$K$ non-zero entries directly from the CSR representation without allocating dense memory.

---

### 12. Multiple $O(N)$ Pandas Anti-Patterns Across 70M Pairs
* **Locations:**
  - [`src/blocking.py#L127`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L127): `self.target_df.iloc[target_row_idx]["entity_id"]`
  - [`src/feature_extraction.py#L126-L127`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py#L126-L127): `target_preprocessed_df.set_index("entity_id").to_dict(orient="index")`
  - [`src/feature_extraction.py#L132`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py#L132): `for _, row in candidates_df.iterrows():`
  - [`src/blocking.py#L154`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L154): `if cand not in cand_map[s1]:`
* **What’s wrong:**
  1. At $K=40$, `candidates_df` contains $1,732,544 \times 40 \approx \mathbf{69,300,000\text{ rows}}$.
  2. `self.target_df.iloc[...]` called 70 million times inside the candidate loop: in pandas, `.iloc` has Python overhead of ~15–20 microseconds per call:
     $$70,000,000 \times 18\,\mu\text{s} \approx 1,260 \text{ seconds} \approx \mathbf{21\text{ minutes just looking up target IDs}}.$$
  3. `target_preprocessed_df.to_dict(orient="index")` on 10 million rows creates 10 million nested Python dicts, consuming ~18 GB of Python heap.
  4. `for _, row in candidates_df.iterrows():` on 70 million rows creates 70 million `pd.Series` wrappers. In pandas, `iterrows()` runs at ~1,500 rows/second:
     $$\frac{69,300,000}{1,500} \approx 46,200 \text{ seconds} \approx \mathbf{12.8\text{ HOURS just to iterate}}.$$
  5. `cand_map[s1]` is a Python `list`. In `format_candidate_pairs_tsv`, `if cand not in cand_map[s1]:` performs an $O(M)$ linear scan on every insertion.
* **Remediation:**
  - Replace `.iloc` with NumPy arrays: `target_ids = self.target_df["entity_id"].to_numpy()`, then access `target_ids[idx]`.
  - Replace `to_dict(orient="index")` with columnar lookups or columnar NumPy arrays.
  - Process feature extraction using `itertuples(index=False, name=None)` or vectorised batch chunks (`zip(*[df[col].to_numpy() for col in ...])`), running multi-threaded via `joblib.Parallel` or `concurrent.futures`.

---

## Priority 3: Blocking Quality & Recall Ceiling

### 13. Monolithic Concatenation Drowns Out Names with Addresses
* **Location:** [`code/business_entity_resolution/src/preprocessing.py#L140-L142`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/preprocessing.py#L140-L142) & [`code/business_entity_resolution/src/blocking.py#L58-L63`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L58-L63)
* **What’s wrong:** You concatenate:
  ```python
  clean_text = clean_name + " | " + clean_address + " | " + country
  ```
  and build a single global TF-IDF vectorizer over the combined string.
  - Business names are short: 1 to 4 words (e.g. `"Starbucks"`, `"Dr. Reddy's Lab"`).
  - Business addresses are long: 10 to 25 words (e.g. `"Plot 42, Silicon Valley Industrial Estate, Whitefield Main Road, Opposite Shell Petrol Pump, Bengaluru 560066"`).
* **Why it matters:**
  1. The vector norm and term weights are dominated by address tokens. Two unrelated businesses in the same commercial tech park or shopping mall (sharing "Floor 3, Silicon Tech Park, Industrial Area") will have high cosine similarity and crowd out true matches.
  2. If a business name is identical or an abbreviation (e.g. "SBI" vs "State Bank of India"), but the address is partial or missing components (common in India), the monolithic TF-IDF similarity falls near zero and the candidate is never retrieved.
  3. **Blocking recall is the hard mathematical ceiling of your final score.** If a true match is not in `candidate_pairs.tsv`, your matching model cannot score it.
* **Remediation:** Implement **Multi-Pass Disjunctive Blocking**:
  - **Channel 1 (Name-Focused):** BM25 / Char 3-gram TF-IDF on `clean_name` alone (top 20).
  - **Channel 2 (Address + First Name Token):** Fast inverted index on postal code / city + first token of business name.
  - Union candidates from both channels. This guarantees candidates with matching names are never crowded out by long address strings.

---

### 14. Fallback in `blocking.py` Causes Catastrophic Unpartitioned Matrix Construction
* **Location:** [`code/business_entity_resolution/src/blocking.py#L83-L89`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L83-L89)
* **What’s wrong:**
  ```python
  if country not in self.country_indices:
      # Fallback: if country not seen in target, consider all targets
      target_sub_indices = list(range(len(self.target_df)))
      char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1)
      char_mat = char_vec.fit_transform(self.target_df["clean_text"])
  ```
  If an entity arrives with `country == "UNKNOWN"` or a typo, the blocker falls back to fitting a character 3-gram TF-IDF on **all 10 million targets** in unpartitioned memory.
* **Why it matters:** This will consume 100+ GB of RAM and crash the job immediately.
* **Remediation:** If a query country is missing or unmapped, default to matching against the candidate target pool of the query's suspected country (derived from address tokens like PIN code or state) or raise a controlled fallback that does not fit a 10M TF-IDF on the fly.

---

## Priority 4: Matching Model Soundness & Overfitting

### 15. Benchmark Dataset Distractor Deficit (False Sense of Security)
* **Location:** [`scripts/create_benchmark.py#L139-L176`](file:///c:/Users/rvsre/Documents/Amazon%20ML/scripts/create_benchmark.py#L139-L176)
* **What’s wrong:** To build `benchmark_source2` and `benchmark_source3`, the script took the true target IDs for the sampled 50k $S1$ entities and added **only 25,000 randomly sampled distractors**:
  ```python
  elif s2_distractors < 25000 and random.random() < 0.05:
      f_out.write(line)
      s2_distractors += 1
  ```
* **Why it matters:**
  - In this 70k benchmark pool, the candidate space contains virtually **zero hard negatives** (businesses with identical names in different zip codes, or businesses sharing an address with different names). Random distractors are trivial for LightGBM to separate.
  - On the real test set, your blocker searches across **10,000,000 targets**. The number of deceptively similar non-matches (chains, franchises, suites in same buildings) is orders of magnitude higher.
  - Your validation score on `benchmark_dataset` (~0.93+) is **massively optimistic and will experience catastrophic precision collapse on the real test set.**
* **Remediation:** Generate benchmark distractors by running the actual blocker over the entire 1.7M training target set and keeping top-K non-matching candidates (lexical hard negatives). Train your LightGBM on the hard negatives that the blocker actually retrieves.

---

### 16. Negative Weight Inflation Sabotages Static Decision Thresholds
* **Location:** [`code/business_entity_resolution/src/models_gbdt.py#L104`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/models_gbdt.py#L104)
* **What’s wrong:** In round 2 of hard-negative mining:
  ```python
  current_sample_weights[hard_fp_mask] *= 2.5
  ```
  Multiplying negative sample weights by 2.5 shifts the underlying class prior that LightGBM optimizes.
  According to probability theory, if negative weights are increased by $w_n$, the predicted probability $\hat{p}$ is related to the true posterior $p$ by:
  $$\hat{p} = \frac{p}{p + w_n(1 - p)}$$
  For $w_n = 2.5$, a true match with true probability $p = 0.70$ will now output:
  $$\hat{p} = \frac{0.70}{0.70 + 2.5(0.30)} = \frac{0.70}{1.45} \approx \mathbf{0.482}$$
* **Why it matters:**
  Your postprocessing config sets `match_threshold = 0.55` and `singleton_threshold = 0.50` (or 0.70 in `postprocessing.py`).
  Because the probabilities have been depressed below 0.50 by the artificial sample weights, **genuine matches will be discarded as singletons**, severely dropping recall.
* **Remediation:**
  If you upweight negatives during training, either:
  1. Recalibrate probabilities on validation data using Isotonic Regression or Platt scaling before thresholding; OR
  2. Perform threshold optimization directly on the out-of-fold predictions produced by the reweighted model rather than using default thresholds like 0.50 / 0.55.

---

## Priority 5: Compliance & Model Licensing Audit

### 17. Hugging Face Network Calls in Evaluation Environment
* **Location:** [`code/business_entity_resolution/src/blocking.py#L31`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L31) and [`code/business_entity_resolution/src/cascade_reranker.py#L27`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/cascade_reranker.py#L27)
* **What’s wrong:**
  ```python
  self.dense_model = SentenceTransformer("BAAI/bge-m3")
  self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
  ```
  Both lines pass Hugging Face model IDs rather than local paths.
* **Why it matters:**
  1. Competition runners in Kaggle / Amazon ML Challenge environments typically have **no internet access** during the private test evaluation. Uncached downloads will throw a network socket error and abort.
  2. If internet is monitored, outbound requests to external servers risk disqualification under the **"STRICTLY PROHIBITED: External Data Lookup"** clause.
* **Remediation:**
  If any transformer model is retained, pre-download its weights to a local `models/` directory inside your package and load via `Path(__file__).parent / "models" / "model_name"`. Confirm all models are licensed under MIT or Apache 2.0 (BGE-M3 is MIT, MiniLM is Apache 2.0; parameter counts $\le 567$M, satisfying $\le 8$B params).

---

## Comparison: Current Pipeline vs. 2025–2026 Industry Standard

| Pipeline Stage | Current Implementation | 2025–2026 Industry SOTA | Why SOTA Wins / Where to Apply in Your Code |
| :--- | :--- | :--- | :--- |
| **Blocking Strategy** | Monolithic character/word TF-IDF on concatenated `name \| addr \| country`. | **Multi-Pass Disjunctive Inverted Blocking** with attribute isolation (Name BM25 + Geo/PIN inverted index). | Addresses drown names in monolithic strings. Disjunctive blocking in [`src/blocking.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py) guarantees candidate generation even when addresses are noisy. |
| **Sparse Retrieval Scaling** | `(b @ mat.T).toarray()` with NumPy dense argpartition ($O(Q \times N)$ memory). | **Direct Sparse Top-K** (`sparse_dot_topn` or CSR-level max-heap). | Eliminates 16 GB dense allocation per batch in [`src/blocking.py#L114`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/blocking.py#L114). Runs in constant memory ($<500$ MB). |
| **Dense Embeddings** | Hardcoded placeholder (`dense_rank = 1000`). | **Matryoshka 2D Embeddings** (e.g. `nomic-embed-text-v1.5` or `bge-small-en-v1.5` truncated to 128 dims) with 8-bit quantization. | 1024-dim BGE-M3 requires ~41 GB RAM for 10M targets. 128-dim quantized vectors fit within ~2.5 GB RAM. |
| **Reranker Architecture** | Off-the-shelf English web search cross-encoder (`ms-marco-MiniLM-L-6-v2`) with sigmoid blending. | **Domain-Trained Tabular GBDT** with token-interaction features or **DeBERTa-v3-small** fine-tuned on entity pairs. | Cross-encoder on 7M pairs on CPU takes 40+ hours. A feature-rich GBDT computes inference in $<90$ seconds. |
| **$F_{0.5}$ Optimization** | Static thresholds ($\tau=0.70$) with no-op margin condition. | **Cost-Sensitive Calibrated Gating** with Isotonic Regression and Bayesian threshold optimization. | $F_{0.5}$ penalizes false positives $2\times$ over false negatives. In [`src/postprocessing.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/postprocessing.py), use a high barrier for singletons ($\tau_s \approx 0.72$) and a strict relative gap ($\Delta \le 0.10$). |
| **Scalable Data Processing** | Pure Python loops with pandas `.iterrows()` and `.iloc`. | **Columnar Vectorization** (`polars` or NumPy structured arrays) + batch streaming. | Reduces feature extraction in [`src/feature_extraction.py`](file:///c:/Users/rvsre/Documents/Amazon%20ML/code/business_entity_resolution/src/feature_extraction.py) from 13 hours down to 10 minutes. |

---

## Actionable Remediation Plan

1. **Delete Dead & Ineffective Code:**
   - Drop `cascade_reranker.py` (removes the 40-hour CPU cross-encoder bottleneck and invalid English MS-MARCO model).
   - In `blocking.py`, remove the dummy `dense_model` / `BAAI/bge-m3` loading to keep the pipeline lean and eliminate external network vulnerabilities.

2. **Fix `blocking.py` Memory & Scaling:**
   - Replace `(b_char @ char_mat.T).toarray()` with sparse top-K retrieval directly from CSR sparse matrices.
   - Replace `self.target_df.iloc[...]` with indexed NumPy arrays:
     ```python
     target_entity_ids = self.target_df["entity_id"].to_numpy()
     cand_id = target_entity_ids[target_row_idx]
     ```
   - Split blocking into two channels: (1) `clean_name` TF-IDF, and (2) `clean_address` TF-IDF, then merge candidate sets.

3. **Fix `preprocessing.py` & `feature_extraction.py`:**
   - Condition diacritics and address replacements on country so French rules don't corrupt US/Indian addresses.
   - Do not sort numbers in `extract_numeric_tokens`; preserve document order so `lead_num_match` actually compares the leading street numbers.
   - Vectorize `build_features_dataframe` using `itertuples(name=None)` and pre-extracted columnar arrays instead of `.iterrows()`.

4. **Fix $F_{0.5}$ Postprocessing & Margin Logic:**
   - Replace the inverted `(max_prob - p) <= (1.0 - min_margin)` condition with a real relative margin `(max_prob - p) <= delta_margin`.
   - Ensure `tau_singleton` acts as the primary barrier ($\ge 0.70$) to prevent false merges on singletons.

5. **Create `code/business_entity_resolution/run_pipeline.py`:**
   - Write the missing end-to-end entrypoint script so the pipeline can be executed and audited with a single command:
     ```bash
     python run_pipeline.py --team-name <team_name>
     ```
   - Verify that running this script produces valid `matching_results.tsv` (1,732,544 rows) and `candidate_pairs.tsv` that pass `student_resource/utils/validate_submission.py` with exit code 0.
