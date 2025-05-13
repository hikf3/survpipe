# run_all_outcomes.py
from evaluation import evaluate_models

# Common inputs
data_path = "data/ML_feat.csv"
output_dir = "/results"
eval_times = [1, 3, 5, 8, 10]

features = [  # combo "f"
    'hypertension', 'substance_abuse', 'nicotine_dependence', 'T2D', 'GERD', 'hyperlipidemia', 'obesity',
    'DAA', 'metformin', 'insulin', 'other_dm_drugs', 'antihypertension', 'anticholesterol', 'antibiotic', 'antigerd',
    'sex_at_birth_code', 'race_code', 'ethnicity_code', 'age_at_hcv_diagnosis',
    'hdl_median', 'hdl_q1', 'hdl_q3', 'ldl_median', 'ldl_q1', 'ldl_q3',
    'a1c_median', 'a1c_q1', 'a1c_q3', 'tg_median', 'tg_q1', 'tg_q3',
    'sbp_median', 'sbp_q1', 'sbp_q3', 'dbp_median', 'dbp_q1', 'dbp_q3',
    'deprivation_index_scaled', 'sdoh_cluster', 'alcohol_use_level_coded'
]

# List of outcomes
outcomes = {
    "cirrhosis": ("years_to_cirrhosis", "has_cirrhosis"),
    "liver_cancer": ("years_to_liver_cancer", "has_liver_cancer"),
    "ascites": ("years_to_ascites", "has_ascites"),
    "encephalopathy": ("years_to_encephalopathy", "has_encephalopathy")
}

# Run each outcome
for outcome_name, (time_col, event_col) in outcomes.items():
    evaluate_models(
        data_path=data_path,
        outcome_name=outcome_name,
        time_col=time_col,
        event_col=event_col,
        features=features,
        eval_times=eval_times,
        output_dir=output_dir
    )
