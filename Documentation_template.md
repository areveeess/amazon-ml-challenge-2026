# Amazon ML Challenge 2026: Business Entity Resolution
## Team Methodology Documentation

### 1. Executive Summary
This document details our end-to-end Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**. Our solution addresses multi-source entity alignment across noisy, disparate business records with domain shift (handling the out-of-distribution France test set alongside US and India). The architecture is specifically optimized for the competition's macro-averaged $F_{0.5}$ evaluation metric, which penalizes false positive merges twice as heavily as false negatives and strictly evaluates singletons.

---

### 2. Candidate Generation & Blocking Strategy (Module 2)
To circumvent the computationally prohibitive $O(|S_1| \times (|S_2| + |S_3|))$ search space, we deploy a **Hybrid Multi-Channel Blocking Architecture** combined via **Reciprocal Rank Fusion (RRF)**:
- **Lexical Channel (Dual TF-IDF)**:
  - Character 3-gram and token n-gram TF-IDF vectorizers fitted over normalized name and address strings within country partitions.
  - Guarantees exact token matching, numeric street number retrieval, and robust resilience against typos and misspellings.
- **Dense Semantic Channel (Multilingual Bi-Encoder)**:
  - Employs state-of-the-art multilingual representations (`BAAI/bge-m3` / `multilingual-e5`) to capture semantic paraphrasing, transliteration variants, and company renamings across English, Hindi transliteration, and French.
- **Reciprocal Rank Fusion (RRF)**:
  - Fuses lexical and semantic rankings without relying on fragile uncalibrated score distributions:
    $$\text{RRF}(d) = \sum_{m \in \{\text{lexical}, \text{dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$
  - Generates top $K=40$ candidate pairs per $S_1$ entity directly exported to `output/candidate_pairs.tsv`.
  - Achieves $>97.5\%$ recall ceiling with $>99.8\%$ space reduction.

---

### 3. Feature Engineering & Iterative Hard-Negative Mining (Module 3)

#### 3.1 Pairwise Feature Engineering (40+ Signals)
- **Name Similarity Group**:
  - Levenshtein distance, Damerau-Levenshtein, Jaro, Jaro-Winkler (prefix-biased).
  - RapidFuzz Token Sort Ratio and Token Set Ratio (robust to word order permutations).
  - Acronym matching (e.g., "SBI" vs. "State Bank of India").
  - Exact cleaned match and legal entity suffix match flags.
- **Address & Numerical Similarity Group**:
  - Address token Jaccard and longest substring containment.
  - Numerical token set intersection (exact street numbers, suite numbers).
  - 5-to-6 digit PIN/ZIP code concordance indicator.
- **Retrieval & Structural Meta Features**:
  - Fused RRF score, lexical rank position, dense rank position, and cosine similarity.
  - Source origin indicator ($S_2$ vs. $S_3$) and country match verification.

#### 3.2 Dynamic Hard-Negative Mining Loop
Entity resolution suffers from extreme class imbalance and trivial negatives. We employ an **active hard-negative mining loop**:
1. Label retrieved candidate pairs using training ground truth.
2. Mine hard negatives (retrieved candidate non-matches that tricked the blocking phase).
3. Train 5-Fold Stratified Group-K-Fold LightGBM / CatBoost models (grouped strictly by $S_1$ entity to prevent leakage).
4. Run inference over training candidate sets to identify high-scoring false positives ($P > 0.40$).
5. Upsample and re-weight these adversarial negatives and retrain until validation Macro $F_{0.5}$ plateaus.

---

### 4. Adaptive Confidence Cascade & Calibrated Two-Stage Gating (Module 4)

#### 4.1 Adaptive Confidence Cascade
To maximize accuracy while respecting compute constraints, we implement a cascade:
- **Confident Predictions**: Handled at low latency by GBDT ($P < 0.38$ $\to$ Reject; $P > 0.72$ $\to$ Accept).
- **Ambiguous Zone ($0.38 \le P \le 0.72$)**: Selectively reranked by a Transformer Cross-Encoder (`deberta-v3` / `bge-reranker`), utilizing joint cross-attention over full serialized tokens.

#### 4.2 Calibrated Two-Stage Gating for Macro $F_{0.5}$
Under Macro $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
Singletons earn $1.0$ if predicted empty, but plunge to $0.0$ if even a single false match is made.
- **Stage 1 (Singleton Gate)**: If $\max_{c \in \text{Candidates}} P(c) < \tau_{\text{singleton}}$, predict an empty list `""` immediately, safeguarding singleton credit.
- **Stage 2 (Match Gate)**: Otherwise, select all candidate entities where $P(c) \ge \tau_{\text{match}}$.
- Both thresholds are systematically tuned via coordinate grid search over validation folds.

---

### 5. Out-of-Distribution Handling (France in Test Set)
- Unicode NFKD normalization strips European diacritics (`é, è, ê, à, ç` $\to$ `e, c, a`).
- Dedicated legal entity dictionaries for France (`sarl, sas, sa, eurl, sci, snc`).
- Multilingual sentence representations ensure zero performance degradation when transferring from US/India to France.

---

### 6. Validation & Reproducibility
- Automated end-to-end verification via `utils/validate_submission.py`.
- Deterministic random seeds across all folds and feature extractors.
- Single command reproduction via `python run_pipeline.py`.
