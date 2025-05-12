# config.py

import os

# Get base directory of the survpipe project
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Path to dataset (expects dataset inside the 'data/' folder)
DATA_PATH = os.path.join(BASE_DIR, "data", "ML_feat.csv")

# Output results directory
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define outcome mappings
SURVIVAL_TARGETS = {
    "cirrhosis": ("years_to_cirrhosis", "has_cirrhosis"),
    "liver_cancer": ("years_to_liver_cancer", "has_liver_cancer"),
    "ascites": ("years_to_ascites", "has_ascites"),
    "encephalopathy": ("years_to_encephalopathy", "has_encephalopathy")
}

# Feature sets
FEATURE_SETS = {
    "a_DX_only": ['hypertension', 'substance_abuse', 'T2D', 'GERD', 'hyperlipidemia', 'obesity'],
    "b_DX_Meds": ['DAA', 'metformin', 'insulin', 'other_dm_drugs', 'antihypertension', 'anticholesterol', 'antibiotic', 'antigerd', 'DAA_dexposure'],
    "c_Demo": ['sex_at_birth_code', 'race_code', 'ethnicity_code', 'age_at_hcv_diagnosis'],
    "d_Labs": ['hdl_median', 'hdl_q1', 'hdl_q3', 'ldl_median', 'ldl_q1', 'ldl_q3', 'a1c_median', 'a1c_q1', 'a1c_q3', 'tg_median', 'tg_q1', 'tg_q3',
                'sbp_median', 'sbp_q1', 'sbp_q3', 'dbp_median', 'dbp_q1', 'dbp_q3', "albumin_median", "albumin_q1", "albumin_q3",
                "bilirubin_median", "bilirubin_q1", "bilirubin_q3", "alt_median", "alt_q1", "alt_q3",
                "ast_median", "ast_q1", "ast_q3", "alp_median", "alp_q1", "alp_q3"],
    "e_Deprivation": ['deprivation_index_scaled'],
    "f_SDoH": ['sdoh_cluster'],
    "g_Lifestyle": ['nicotine_dependence', 'alcohol_use_level_coded']
}

# Feature combinations
COMBO_SETS = {
    "a": FEATURE_SETS["a_DX_only"],
    "b": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"],
    "c": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"],
    "d": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"] + FEATURE_SETS["d_Labs"],
    "e": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"] + FEATURE_SETS["d_Labs"] + FEATURE_SETS["g_Lifestyle"],
    "f": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"] + FEATURE_SETS["d_Labs"] + FEATURE_SETS["g_Lifestyle"] + FEATURE_SETS["e_Deprivation"],
    "g": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"] + FEATURE_SETS["d_Labs"] + FEATURE_SETS["g_Lifestyle"] + FEATURE_SETS["f_SDoH"],
    "h": FEATURE_SETS["a_DX_only"] + FEATURE_SETS["b_DX_Meds"] + FEATURE_SETS["c_Demo"] + FEATURE_SETS["d_Labs"] + FEATURE_SETS["g_Lifestyle"] + FEATURE_SETS["f_SDoH"] + FEATURE_SETS["e_Deprivation"]
}
