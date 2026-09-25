# Business Entity Resolution Pipeline (Amazon ML Challenge 2026)

This repository contains the complete, reproducible end-to-end Machine Learning pipeline for the Amazon ML Challenge 2026.

## Architecture Overview
The system is partitioned into 4 decoupled, production-grade modules:
1. **Module 1: Preprocessing & Normalization Engine (`src/preprocessing.py`)**
   - Unicode NFKD diacritic normalization (handles French test entities).
   - Multi-country corporate suffix stripping (US, India, France).
   - Address canonicalization and numerical PIN/building token extraction.
2. **Module 2: High-Recall Hybrid Blocking (`src/blocking.py`)**
   - Dual-channel candidate retrieval: Character 3-gram TF-IDF + Multilingual Dense Embeddings.
   - Reciprocal Rank Fusion (RRF) for robust rank-based candidate merging.
   - Generates competition candidate set (`output/candidate_pairs.tsv`).
3. **Module 3: Feature Engineering & Dynamic Hard-Negative GBDT (`src/feature_extraction.py`, `src/models_gbdt.py`)**
   - Computes 40+ pairwise string, token, numerical, and retrieval features.
   - 5-Fold Stratified Group-K-Fold LightGBM classifier.
   - Dynamic Hard-Negative Mining loop actively retraining on deceptive false positives.
4. **Module 4: Cascade Reranker, Two-Stage Gating & Packaging (`src/cascade_reranker.py`, `src/evaluate.py`, `src/postprocessing.py`, `src/package_submission.py`)**
   - Adaptive Confidence Cascade for ambiguous probability pairs.
   - Calibrated Two-Stage Gating specifically tuned for Macro $F_{0.5}$ and singleton preservation.
   - Formatter for `output/matching_results.tsv` and automated submission packaging into `<team_name>_submission.zip`.

## Installation & Setup
```bash
pip install -r requirements.txt
```

## Running End-to-End Pipeline
Place the competition dataset in `dataset/train/` and `dataset/test/`, then run:
```bash
python run_pipeline.py --team-name <your_team_name>
```

## Validation & Formatting Check
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
