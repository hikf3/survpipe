# survpipe: Time-to-Event Machine Learning Pipeline

`survpipe` is a modular Python pipeline for training and evaluating survival (time-to-event) models using real-world health data. It supports cross-validated evaluation of multiple models and feature sets across various clinical outcomes.

---

A reproducible survival-analysis pipeline for tabular EHR-style data with:
- leakage-safe preprocessing fit on TRAIN only
- K-fold CV hyperparameter selection
- time-dependent AUC at multiple horizons
- bootstrap C-index on test
- permutation-importance feature ranking (raw features)
- reduced models (Top 50% / Top 25%) based on CV permutation importance
- optional Kaplan–Meier plots and log-rank tests (if `lifelines` is installed)

Supported model families:
- Coxnet Elastic Net (`Coxnet-EN`)
- Coxnet LASSO (`Coxnet-LASSO`)
- Random Survival Forest (`RSF`)
- Gradient Boosted Survival (`GBSA`)
- Cox Neural Network (`CoxNN`, PyTorch)

> Note: `scikit-survival` is required for Coxnet/RSF/GBSA and survival metrics. `PyTorch` is required only for `CoxNN`.

---

## Repository structure (recommended)


## 📁 Project Structure

```
├── surv2/
│ ├── init.py
│ ├── config.py
│ ├── preprocess.py
│ ├── models.py
│ ├── utils_surv.py
│ ├── pipeline_survival.py
│ └── run_all.py
├── splits/
│ ├── cirrhosis_split_train_idx.csv
│ ├── cirrhosis_split_test_idx.csv
│ ├── HCC_split_train_idx.csv
│ ├── HCC_split_test_idx.csv
│ ├── death_split_train_idx.csv
│ └── death_split_test_idx.csv
├── hcv_original_feat.csv
├── requirements.txt
└── README.md
```


Your `splits/*.csv` files must contain **row indices** (0-based) with **no header**.

---

Time units: The pipeline assumes time is in years (recommended).

## Preprocessing (Leakage-Safe)

All preprocessing is fit on the **TRAIN split only** to prevent data leakage.

### Categorical Features
- Constant fill with `"__MISSING__"`
- One-hot encoding

### Binary / Ordinal Features
- Numeric coercion
- Constant fill with `0`

### Numeric / Genotype Features
- Median imputation
- Signed `log1p` transform for features where `|skew| > SKEW_THRESH`
- RMS scaling (divide by root mean square)
- Add missingness indicators for mid/high-missing columns

### Missingness & Skew Thresholds

Controlled in `config.py`:
- `LOW_MISSING_THRESH`
- `HIGH_MISSING_THRESH`
- `SKEW_THRESH`

---

## What the Pipeline Runs

For each outcome:

1. Load dataset from `DATA_PATH`
2. Load split indices from `SPLIT_DIR`
3. Clean invalid rows:
   - Non-finite or non-positive survival time
   - Non-finite event indicator

For each model in `MODEL_NAMES`:

### Cross-Validation Phase
- Grid search for best hyperparameters
- Cross-validation permutation importance (raw features)
- Save reduced feature lists:
  - Top 50%
  - Top 25%

### Final Training
- Fit final model on entire TRAIN split

### Test Evaluation
- C-index + bootstrap confidence interval
- Time-dependent AUC at horizons (default: 3, 5, 10 years)
- Brier score at 5 years (complete-case)
- IPCW Brier score (if survival function available)
- Logistic calibration slope & intercept (when horizon risk available)
- Save test predictions CSV
- Test permutation importance (Top-K features, OOM-safe)
- ROC curve JSONs (3, 5, 10 years)
- Kaplan–Meier plot PNG + JSON (if `lifelines` installed)

---

## Outputs

All outputs are written to: OUTDIR/outcome=<OUTCOME>/


### Cross-Validation Outputs
- `cv/grid_metrics_<setting>_<model>.csv`
- `cv/cv_metrics_<setting>_<model>.csv`
- `cv/cv_perm_importance_mean_<setting>_<model>.csv`
- `cv/reduced_lists/<model>_top50.json`
- `cv/reduced_lists/<model>_top25.json`

### Final Model Artifacts
final_models/<setting>/<model>/

- `model.joblib`
- `preprocessor.joblib`
- `features_raw.json`
- `training_manifest.json`

### Test Outputs
- `test/test_metrics_<setting>_<model>.csv`
- `test/test_predictions_<setting>_<model>.csv`

### Plots
- `plots/roc_<setting>_<model>_<horizon>yr.json`
- `plots/km_<setting>_<model>_5yr.png` (if `lifelines` available)

---

## Model Settings

`<setting>` is one of:
- `full`
- `top50`
- `top25`

---

## Quickstart

### 1) Create Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
---
### 2) Configure Paths and Columns
```
Edit surv2/config.py:

DATA_PATH — dataset CSV

SPLIT_DIR + split filename templates

OUTCOMES time/event column names

Feature lists and thresholds
```
### 3) Run the Pipeline
```python -m surv2.run_all```

---
## `requirements.txt`

```
# Core
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
joblib>=1.2

# Survival models + metrics (required for Coxnet/RSF/GBSA + tdAUC + IPCW Brier)
scikit-survival>=0.22

# Plots / posthoc KM + logrank (optional but recommended)
lifelines>=0.27
matplotlib>=3.7

# Neural net Cox model (optional; only needed if using model "CoxNN")
torch>=2.0
```
