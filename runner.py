# runner.py
import os
import numpy as np
import pandas as pd
import time
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv

from config import SURVIVAL_TARGETS, COMBO_SETS, RESULTS_DIR
from models import GET_MODEL_SET, GET_PARAM_GRIDS

def run_model(model_name, outcome_name, data, output_path=RESULTS_DIR):
    
    os.makedirs(output_path, exist_ok=True)
    results = []

    start_time = time.time()

    time_col, event_col = SURVIVAL_TARGETS[outcome_name]
    grid = GET_PARAM_GRIDS().get(model_name, {})
    param_combos = [dict(zip(grid.keys(), v)) for v in product(*grid.values())] if grid else [{}]
    model_fn = GET_MODEL_SET()[model_name]


    print(f"\n🔍 Model: {model_name} | Outcome: {outcome_name}")

    for feat_key, features in COMBO_SETS.items():
        df = data.dropna(subset=[time_col] + features)
        df = df[df[event_col].isin([0, 1])]
        if df.empty or df[event_col].nunique() < 2:
            print(f"⚠ Skipping {feat_key} due to insufficient variation")
            continue

        y = Surv.from_dataframe(event=event_col, time=time_col, data=df)
        X = df[features]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for p in param_combos:
            cindices = []
            
            for train_idx, test_idx in skf.split(X, df[event_col]):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                try:
                    model = model_fn(p)
                    model.fit(X_train, y_train)                 
                    # Score metrics
                    cindex = model.score(X_test, y_test) 
                    cindices.append(cindex)
           
                except Exception as e:
                    print(f"❌ {feat_key} | Params={p} | Error: {e}")
                    continue

            if cindices:
                results.append({
                    "Model": model_name,
                    "Outcome": outcome_name,
                    "Feature_Set": feat_key,
                    "Params": str(p),
                    "Mean_Cindex": round(np.mean(cindices), 4),
                    "Std_Cindex": round(np.std(cindices), 4)            
                })

        print(f"✅ Done with feature set: {feat_key}")
        print("="*50)

    result_df = pd.DataFrame(results)
    out_csv = os.path.join(output_path, f"{model_name}_{outcome_name}_gridsearch.csv")
    result_df.to_csv(out_csv, index=False)
    
    total_time = round(time.time() - start_time, 2)
    total_minutes = round(total_time / 60, 2)
    
    print(f"💾 Saved to: {out_csv}")
    print(f"⏱ Total runtime: {total_time} seconds ({total_minutes} minutes)")
