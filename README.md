# survpipe: Time-to-Event Machine Learning Pipeline

`survpipe` is a modular Python pipeline for training and evaluating survival (time-to-event) models using real-world health data. It supports cross-validated evaluation of multiple models and feature sets across various clinical outcomes.

---

## 📁 Project Structure

```
survpipe/
│
├── data/
│ └── ML_feat.csv # Input dataset
│
├── results/ # Output directory for results
│
├── config.py # Configuration for paths, outcomes, and features
├── models.py # Model definitions with hyperparameter grids
├── runner.py # Core function to run models with 5-fold CV
└── main.py # Entry point script to execute training and evaluation
```

## 🔍 Features

- Supports survival models:  
  - Random Survival Forest (`RSF`)  
  - Gradient Boosting Survival Analysis (`GBSA`)  
  - Coxnet (Lasso/ElasticNet penalized Cox regression)

- Evaluates across multiple:
  - **Outcomes** (e.g., cirrhosis, liver cancer)
  - **Feature sets** (diagnoses, meds, labs, lifestyle, etc.)

- Cross-validation metrics:
  - **C-index**
  - **Integrated Brier Score (IBS)**

- Automatically handles:
  - Dataset loading
  - Folder creation for results

---

## 🚀 Quick Start

1. **Place your dataset** in the `data/` directory and rename it to `ML_feat.csv` (or modify `config.py` to reflect a different name).

2. **Install dependencies** (recommended in a virtual environment):
   ```bash
   pip install -r requirements.txt
   
3. **Run the pipeline**
    ```bash
    python3 main.py
    
4. **Results** will be saved in results/ as CSV files (one per model per outcome)

## ⚙️ Customization
-To add a new model, define it in models.py and include it in the `GET_MODEL_SET` dictionary.
-To add new parameter combinations, define them in models.py and include them in the `GET_PARAM_GRIDS` dictionary.

-To change features or outcomes, edit `config.py`.

-To modify evaluation, update `runner.py`.

## 📊 Output Files
**Each CSV in the results/ directory includes**:

  -Model and outcome name
  
  -Feature set key
  
  -Parameter configuration
  
  -Mean and standard deviation of:
  
    - **C-index**
  
    - **Integrated Brier Score (IBS)**
