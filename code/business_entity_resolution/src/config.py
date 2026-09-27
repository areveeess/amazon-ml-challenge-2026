"""
Configuration and Global Paths for Business Entity Resolution Pipeline.
"""
from dataclasses import dataclass
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Auto-detect real dataset location: prefers student_resource/dataset if present
if (PROJECT_ROOT / "student_resource" / "dataset" / "train").exists():
    DATASET_DIR = PROJECT_ROOT / "student_resource" / "dataset"
else:
    DATASET_DIR = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

MATCHING_RESULTS_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

@dataclass
class BlockingConfig:
    top_k_candidates: int = 40
    lexical_top_k: int = 35
    dense_top_k: int = 35
    rrf_k: int = 60
    embedding_model_name: str = "BAAI/bge-m3"  # Multilingual SOTA
    fallback_embedding_model: str = "intfloat/multilingual-e5-base"
    batch_size: int = 64
    train_s1_sample_size: int = 50000
    train_target_sample_size: int = 250000

@dataclass
class ModelConfig:
    n_splits: int = 5
    seed: int = 42
    use_gpu: bool = True  # Enable GPU-accelerated GBDT (CatBoost GPU / PyTorch CUDA)
    hard_negative_ratio: int = 8
    max_hard_neg_mining_rounds: int = 2
    hard_neg_percentile: float = 90.0     # Mine top X% of negative score distribution
    max_sample_weight: float = 10.0       # Cap on per-sample weight after escalation
    catboost_params = {
        "iterations": 600,
        "learning_rate": 0.08,
        "depth": 6,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "task_type": "GPU",
        "verbose": False,
        "early_stopping_rounds": 40
    }
    lgb_params = {
        "objective": "binary",
        "metric": "None",  # Custom F0.5 feval used for early stopping
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 800,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    }

@dataclass
class CascadeConfig:
    ambiguity_low: float = 0.38
    ambiguity_high: float = 0.72
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32

@dataclass
class GatingConfig:
    # Tuned via Optuna study with true relative delta_margin.
    singleton_threshold: float = 0.75  # If max_prob < threshold, declare singleton (empty list)
    match_threshold: float = 0.90      # If non-singleton, accept pairs with prob >= threshold
    delta_margin: float = 0.11         # Relative confidence gap: (max_p - p) <= delta_margin
    min_margin: float = 0.11           # Alias for delta_margin for backward compatibility
