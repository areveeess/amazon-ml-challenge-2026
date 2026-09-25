# Amazon ML Challenge 2026 - Business Entity Resolution Pipeline

High-performance, scalable entity resolution pipeline for multi-source business record matching under heavy noise, non-standard naming variations, and international address variations.

## Problem Overview
Given business records from three independent data sources (Source 1 reference, Source 2, and Source 3), the objective is to match each Source 1 business entity with its corresponding records in Source 2 and Source 3, accounting for:
- Severe name variations, abbreviations, legal suffixes, and noise.
- Multi-lingual and non-standard address formats across US, India, and France.
- Singletons (entities with zero matches in sources 2/3).
- Scale: Millions of entity pairs requiring fast candidate retrieval (blocking) and precision re-ranking.

## Repository Structure
```
├── code/                         # Core entity resolution pipeline modules
├── scripts/                      # Utility scripts (benchmark creation, EDA, hyperparameter tuning)
│   ├── create_benchmark.py       # Representative stratified 50k benchmark generator
│   ├── explore_data.py           # Dataset exploration and stats analysis
│   └── tune_study.py             # Optuna/hyperparameter search routines
├── utils/                        # Shared helper functions
├── benchmark_dataset/            # Stratified 50k benchmark subset for rapid experimentation
├── student_resource/             # Official challenge documentation and validation utils
├── run_pipeline.py               # Main pipeline execution entry point
├── tuning_results.csv            # Empirical validation and parameter tuning results
├── ENTITY_RESOLUTION_PIPELINE_CRITIQUE.md # Architecture critique and optimization insights
├── Documentation_template.md     # Official challenge documentation submission template
└── .gitignore                    # Git ignore file (excludes virtual environments and raw data)
```

## Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or install required dependencies (pandas, scikit-learn, catboost, etc.)
```

### 2. Running on Benchmark Dataset
To quickly test and evaluate the pipeline on the sample benchmark dataset:
```bash
python run_pipeline.py --benchmark
```

### 3. Running Full Pipeline
Place full competition datasets inside `student_resource/dataset/train/` and `student_resource/dataset/test/`, then execute:
```bash
python run_pipeline.py
```

## Architecture Highlights
- **Multi-Stage Candidate Retrieval (Blocking)**: Combines token n-gram inversion, prefix blocking, and country-constrained candidate pooling.
- **Hybrid Feature Engineering**: Jaro-Winkler, Levenshtein, Token Sort/Set ratios, address component overlap, and phonetic encoding.
- **Gradient Boosted Ranking & Calibrated Thresholding**: High-precision classification paired with threshold optimization to maximize Macro F1 score while avoiding false merges.
