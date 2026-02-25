# surv2/config.py
from __future__ import annotations

from sklearn.model_selection import ParameterGrid

# ====================
# I/O
# ====================
DATA_PATH = "hcv_original_feat.csv"

SPLIT_DIR = "splits"
TRAIN_IDS_TEMPLATE = "{outcome}_split_train_idx.csv"  # row indices, no header
TEST_IDS_TEMPLATE  = "{outcome}_split_test_idx.csv"   # row indices, no header

OUTDIR = "results_survpipe_v2"
SEED = 42

# ====================
# Outcomes (MUST match your CSV)
# ====================
# Units: time in YEARS (recommended). If your time is days, convert before running.
OUTCOMES = {
    "cirrhosis": {"time_col": "time_cirrhosis_min_years", "event_col": "event_cirrhosis_adj"},
    "HCC":       {"time_col": "time_liver_cancer_min_years", "event_col": "event_liver_cancer_adj"},
    "death":     {"time_col": "years_to_death","event_col": "event_death"},
}

# ====================
# CV / Horizons
# ====================
N_SPLITS_CV = 5

HORIZONS_YEARS = [3.0, 5.0, 10.0]
PRIMARY_HORIZON_YEARS = 5.0

# ====================
# Permutation importance controls (OOM safe)
# ====================
PI_REPEATS_CV = 5
PI_REPEATS_TEST = 5
PI_TOPK_TEST = 50            # Only permute top-K features on test

# ====================
# Bootstrap controls (test)
# ====================
BOOTSTRAPS = 1000

# ====================
# Reduced models
# ====================
# From CV mean permutation importance, per model
REDUCED_FRACS = [0.50, 0.25]   # 50% and 25%

# Optional “paper-style” significance filter after reduced set selection:
# if enabled, you can take e.g. top15 then keep only significant by logrank (p<alpha).
ENABLE_LOGRANK_FEATURE_FILTER = False
LOGRANK_TOP_N = 15
LOGRANK_ALPHA = 0.05

# ====================
# Feature lists (UPDATED)
# ====================
BINARY_COLS = [
    "hypertension","substance_abuse","T2D","GERD","hyperlipidemia","obesity",
    "DAA_within_180d","metformin","insulin","other_dm_drugs",
    "antihypertension","anticholesterol","antibiotic","antigerd","smoking_status"
]
CAT_COLS = ["sex_at_birth_clean","race_clean","ethnicity_clean","alcohol_use_level"]
ORDINAL_COLS: list[str] = []

NUM_COLS = [
    "sbp_mean","dbp_mean","bilirubin_mean","alt_mean","ast_mean","alp_mean",
    "albumin_mean","hdl_mean","ldl_mean","tg_mean","a1c_mean",
    "age_at_hcv_diagnosis","deprivation_index"
]

GENO_COLS = [
    "chr19:19308262","chr19:39241143","chr19:39252525","chr19:44908684","chr19:44912921",
    "chr19:19268740","chr19:39248147","chr22:43928847","chr22:43928850","chr14:94378610",
]

RAW_FEATURES = BINARY_COLS + CAT_COLS + ORDINAL_COLS + NUM_COLS + GENO_COLS

# ====================
# Models to run
# ====================
MODEL_NAMES = ["Coxnet-EN", "Coxnet-LASSO", "RSF", "GBSA", "CoxNN"]

# ====================
# Preprocess knobs
# ====================
LOW_MISSING_THRESH = 0.20
HIGH_MISSING_THRESH = 0.50
SKEW_THRESH = 3.0

# ====================
# Tree model resource controls (avoid killed jobs)
# ====================
# You can set RSF_NJOBS=1 or 2 if memory pressure is high.
RSF_NJOBS = -1

# ==============================
# Hyperparameter grids (tuned inside CV)
# ==============================
def get_param_grid(model_name: str):
    if model_name.startswith("Coxnet"):
        return list(ParameterGrid({
            "alpha_min_ratio": [0.01, 0.05],
            "max_iter": [20000],
        }))
    if model_name == "RSF":
        return list(ParameterGrid({
            "n_estimators": [500],
            "min_samples_leaf": [3, 5, 10],
            "min_samples_split": [10],
            "max_features": ["sqrt"],
        }))
    if model_name == "GBSA":
        return list(ParameterGrid({
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "max_depth": [2, 3],
        }))
    if model_name == "CoxNN":
        return list(ParameterGrid({
            "lr": [1e-3],
            "epochs": [250, 300],
            "patience": [25],
            "wd": [1e-4, 5e-4],
            "pdrop": [0.05, 0.10],
        }))
    return [dict()]
