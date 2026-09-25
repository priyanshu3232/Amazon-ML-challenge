"""Central configuration: paths and tunable parameters for the ER pipeline.

Every heavy script reads from here so the same code runs unchanged on a laptop
(dev sample) and on a remote CPU/GPU box (full scale). Override paths with the
ER_ROOT environment variable if the repo is checked out elsewhere.
"""
import os
from pathlib import Path

# Repository root that contains student_resource/, output/, work/
ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[3]))
DATA_DIR = ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "output"          # final submission files
WORK_DIR = ROOT / "work"              # intermediate artefacts (models, parquet)

# ---- blocking -------------------------------------------------------------
BLOCK_TOPK_WORD = 25       # per S1 query: top-K by word TF-IDF (name+address)
BLOCK_TOPK_NAME = 15       # per S1 query: top-K by word TF-IDF (name only)
BLOCK_TOPK_CHAR = 10       # per S1 query: top-K by char-3gram TF-IDF (name only)
BLOCK_TOPK_DOMAIN = 5      # per S1 query: top-K domain-name pool records (char 3-gram on space-less name)
BLOCK_DOMAIN_MIN_SIM = 0.35
BLOCK_TOPK_REVERSE = 3     # per S2/S3 record: top-K S1 by word TF-IDF (name+address)
BLOCK_MIN_SIM = 0.05       # drop candidate pairs below this cosine in forward passes
BLOCK_REV_MIN_SIM = 0.15   # stricter floor for the reverse (pool -> S1) pass
BLOCK_CHAR_MAX_DF = 0.005  # prune ubiquitous char n-grams (speed, little recall loss)
BLOCK_WORD_MAX_DF = 0.02   # prune ubiquitous word tokens (pvt, ltd, rd, big cities)
N_THREADS = max(1, (os.cpu_count() or 2) - 1)

# ---- model ----------------------------------------------------------------
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=N_THREADS,
)
LGB_ROUNDS = 1500
LGB_EARLY_STOP = 100

# ---- validation split -----------------------------------------------------
VAL_FRACTION = 0.2
SEED = 42
