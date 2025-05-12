# models.py

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis

# Dictionary of model constructors
def GET_MODEL_SET():
    return {
        "RSF": lambda p: make_pipeline(
            StandardScaler(),
            RandomSurvivalForest(n_jobs=-1, random_state=42, **p)
        ),
        "GBSA": lambda p: make_pipeline(
            StandardScaler(),
            GradientBoostingSurvivalAnalysis(random_state=42, **p)
        ),
        "Coxnet": lambda p: make_pipeline(
            StandardScaler(),
            CoxnetSurvivalAnalysis(
                l1_ratio=p.get("l1_ratio", 1.0),
                alpha_min_ratio=0.01,
                max_iter=10000
            )
        )
    }

# Dictionary of hyperparameter grids
def GET_PARAM_GRIDS():
    return {
        "RSF": {
            "min_samples_leaf": [5, 15],
            "min_samples_split": [5, 10]
        },
        "GBSA": {
            "learning_rate": [0.1],
            "n_estimators": [300],
            "max_depth": [3]
        },
        "Coxnet": {
            "l1_ratio": [0.2, 0.5, 1.0]
        }
    }
