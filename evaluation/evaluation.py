# evaluation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, brier_score

def predict_surv_array(model, X, times):
    surv_funcs = model.predict_survival_function(X)
    return np.asarray([[sf(t) for t in times] for sf in surv_funcs])

def evaluate_models(
    data_path,
    outcome_name,
    time_col,
    event_col,
    features,
    eval_times,
    output_dir="results"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    df = df.dropna(subset=[time_col, event_col] + features)
    df = df[df[event_col].isin([0, 1])]

    y = Surv.from_dataframe(event=event_col, time=time_col, data=df)
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df[event_col], random_state=42
    )

    results = {}

    # --- RSF ---
    rsf = make_pipeline(
        StandardScaler(),
        RandomSurvivalForest(
            n_estimators=100, min_samples_split=10, min_samples_leaf=15,
            max_features="sqrt", random_state=42
        )
    )
    rsf.fit(X_train, y_train)
    rsf_surv = rsf.predict(X_test)
    rsf_auc, _ = cumulative_dynamic_auc(y_train, y_test, rsf_surv, eval_times)
    rsf_probs = predict_surv_array(rsf, X_test, eval_times)
    _, rsf_brier = brier_score(y_train, y_test, rsf_probs, eval_times)
    results["RSF"] = (rsf_auc, rsf_brier)

    # --- GBSA ---
    gbsa = make_pipeline(
        StandardScaler(),
        GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42
        )
    )
    gbsa.fit(X_train, y_train)
    gbsa_surv = gbsa.predict(X_test)
    gbsa_auc, _ = cumulative_dynamic_auc(y_train, y_test, gbsa_surv, eval_times)
    gbsa_probs = predict_surv_array(gbsa, X_test, eval_times)
    _, gbsa_brier = brier_score(y_train, y_test, gbsa_probs, eval_times)
    results["GBSA"] = (gbsa_auc, gbsa_brier)

    # --- Coxnet ---
    coxnet = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(
            l1_ratio=1.0, alpha_min_ratio=0.01, max_iter=10000, fit_baseline_model=True
        )
    )
    coxnet.fit(X_train, y_train)
    coxnet_surv = coxnet.predict(X_test)
    coxnet_auc, _ = cumulative_dynamic_auc(y_train, y_test, coxnet_surv, eval_times)
    coxnet_probs = predict_surv_array(coxnet, X_test, eval_times)
    _, coxnet_brier = brier_score(y_train, y_test, coxnet_probs, eval_times)
    results["Coxnet"] = (coxnet_auc, coxnet_brier)

    # --- Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for model, (auc, _) in results.items():
        axs[0].plot(eval_times, auc, marker='o', label=f"{model} (AUC={np.mean(auc):.2f})")
    axs[0].set_title("Time-dependent AUC")
    axs[0].set_xlabel("Time (years)")
    axs[0].set_ylabel("AUC")
    axs[0].legend()
    axs[0].grid(True)

    for model, (_, brier) in results.items():
        axs[1].plot(eval_times, brier, marker='s', label=f"{model} (Brier={np.mean(brier):.2f})")
    axs[1].set_title("Brier Score")
    axs[1].set_xlabel("Time (years)")
    axs[1].set_ylabel("Brier Score")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{outcome_name}_auc_brier.png"), dpi=300)

    # --- Save numeric summary ---
    summary_df = pd.DataFrame({
        "Time": eval_times
    })
    for model in results:
        summary_df[f"{model}_AUC"] = results[model][0]
        summary_df[f"{model}_Brier"] = results[model][1]

    summary_df.to_csv(os.path.join(output_dir, f"{outcome_name}_metrics.csv"), index=False)
    print(f"✅ Completed: {outcome_name}")

