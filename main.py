# Content for main.py

# main.py
## main.py

import os
import pandas as pd
from config import DATA_PATH
from runner import run_model

# Load dataset
data = pd.read_csv(DATA_PATH)

# List of (model, outcome) pairs to run
tasks = [
    ("RSF", "cirrhosis"), ("GBSA", "cirrhosis"), ("Coxnet", "cirrhosis"),
    ("RSF", "liver_cancer"), ("GBSA", "liver_cancer"), ("Coxnet", "liver_cancer"),
    ("RSF", "ascites"), ("GBSA", "ascites"), ("Coxnet", "ascites"),
    ("RSF", "encephalopathy"), ("GBSA", "encephalopathy"), ("Coxnet", "encephalopathy")
]

# Run all tasks
for model_name, outcome_name in tasks:
    print(f"🚀 Running {model_name} for {outcome_name}")
    run_model(
        model_name=model_name,
        outcome_name=outcome_name,
        data=data   
    )


