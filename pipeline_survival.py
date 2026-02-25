from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from surv2 import config as C
from surv2.preprocess import build_preprocessor
from surv2.models import make_model
from surv2.utils_surv import (
    log, ensure_dir, save_json,
    make_y_struct, cindex, auc_td_train_test,
    brier_at_horizon_complete_case, brier_ipcw_at_horizon,
    calibrate_logistic,
    bootstrap_cindex, roc_at_horizon,
    permutation_importance_raw, mean_pi_table,
    save_model_bundle, predicted_risk_at_horizon,
    survival_prob_at_t,
    logrank_feature_table, km_plot_low_high,
)

# -------------------------
# Preprocess defaults
# -------------------------
LOW_MISS = float(getattr(C, "LOW_MISSING_THRESH", 0.20))
HIGH_MISS = float(getattr(C, "HIGH_MISSING_THRESH", 0.50))
SKEW_THRESH = float(getattr(C, "SKEW_THRESH", 3.0))


# -------------------------
# Split helpers
# -------------------------
def read_split_indices(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None).iloc[:, 0].to_numpy().astype(int)


def make_train_test(df: pd.DataFrame, outcome: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    tr_path = Path(C.SPLIT_DIR) / C.TRAIN_IDS_TEMPLATE.format(outcome=outcome)
    te_path = Path(C.SPLIT_DIR) / C.TEST_IDS_TEMPLATE.format(outcome=outcome)

    log(f"[SPLIT] reading train={tr_path} test={te_path}")
    train_idx = read_split_indices(tr_path)
    test_idx = read_split_indices(te_path)

    overlap = np.intersect1d(train_idx, test_idx).size
    if overlap != 0:
        raise RuntimeError(f"Train/test split overlap detected: {overlap} indices")

    if train_idx.max() >= len(df) or test_idx.max() >= len(df):
        raise RuntimeError("Split indices out of bounds for dataset length")

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    log(f"[SPLIT] train n={len(df_train)} test n={len(df_test)} overlap=0 ✅")
    return df_train, df_test, train_idx, test_idx


def clean_for_outcome(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    x = df.copy()
    if time_col not in x.columns or event_col not in x.columns:
        raise KeyError(f"Missing outcome columns: time_col={time_col}, event_col={event_col}")

    x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    x[event_col] = pd.to_numeric(x[event_col], errors="coerce")
    before = len(x)

    x = x[np.isfinite(x[time_col].to_numpy())]
    x = x[np.isfinite(x[event_col].to_numpy())]
    x = x[x[time_col] > 0]
    x[event_col] = (x[event_col].astype(int) > 0).astype(int)

    after = len(x)
    log(f"[CLEAN] dropped {before - after} invalid time/event rows; kept n={after}")
    return x


# -------------------------
# Common paths / "done?" checks
# -------------------------
def outcome_dir(outcome: str) -> Path:
    return Path(C.OUTDIR) / f"outcome={outcome}"


def metrics_path(outcome: str, model_name: str, setting: str) -> Path:
    return outcome_dir(outcome) / "test" / f"test_metrics_{setting}_{model_name}.csv"


def reduced_list_path(outcome: str, model_name: str, tag: str) -> Path:
    return outcome_dir(outcome) / "cv" / "reduced_lists" / f"{model_name}_{tag}.json"


def is_done(outcome: str, model_name: str, setting: str) -> bool:
    return metrics_path(outcome, model_name, setting).exists()


# -------------------------
# CV grid search
# -------------------------
def cv_grid_search(model_name: str, X_raw: pd.DataFrame, y_struct, seed: int) -> Tuple[dict, pd.DataFrame]:
    grid = C.get_param_grid(model_name) or [dict()]
    kf = KFold(n_splits=C.N_SPLITS_CV, shuffle=True, random_state=seed)

    rows: List[dict] = []
    for pid, params in enumerate(grid):
        params = dict(params)
        params["random_state"] = seed

        if model_name == "RSF":
            params.setdefault("n_jobs", int(getattr(C, "RSF_NJOBS", 1)))

        log(f"[GRID] model={model_name} paramset={pid + 1}/{len(grid)} params={params}")

        for fold, (i_tr, i_va) in enumerate(kf.split(X_raw), start=1):
            Xtr_raw = X_raw.iloc[i_tr].copy()
            Xva_raw = X_raw.iloc[i_va].copy()
            ytr = y_struct[i_tr]
            yva = y_struct[i_va]

            pre = build_preprocessor(
                Xtr_raw,
                binary_cols=C.BINARY_COLS,
                cat_cols=C.CAT_COLS,
                ordinal_cols=C.ORDINAL_COLS,
                numeric_cols=C.NUM_COLS,
                geno_cols=C.GENO_COLS,
                low_missing_thresh=LOW_MISS,
                high_missing_thresh=HIGH_MISS,
                skew_thresh=SKEW_THRESH,
            )

            Xt = np.asarray(pre.fit_transform(Xtr_raw), dtype=np.float32)
            Xv = np.asarray(pre.transform(Xva_raw), dtype=np.float32)

            m = make_model(model_name, params=params)
            m.fit(Xt, ytr)

            rv = m.predict_risk(Xv)
            fold_c = cindex(yva, rv)
            aucs = auc_td_train_test(ytr, yva, rv, C.HORIZONS_YEARS)

            rows.append({"param_id": pid, "fold": fold, "cindex": fold_c, **aucs})

            del m, pre, Xt, Xv, Xtr_raw, Xva_raw
            gc.collect()

    df_grid = pd.DataFrame(rows)
    if df_grid.empty:
        return {}, df_grid

    cols = ["cindex"] + [f"AUC@{int(h)}yr" for h in C.HORIZONS_YEARS]
    mean_by_param = df_grid.groupby("param_id")[cols].mean()

    best_pid = mean_by_param["cindex"].idxmax()
    ties = mean_by_param.index[mean_by_param["cindex"] == mean_by_param.loc[best_pid, "cindex"]].tolist()
    if len(ties) > 1:
        key = f"AUC@{int(C.PRIMARY_HORIZON_YEARS)}yr"
        if key in mean_by_param.columns:
            best_pid = mean_by_param.loc[ties, key].idxmax()

    best_params = dict(grid[int(best_pid)])
    best_params["random_state"] = seed
    if model_name == "RSF":
        best_params.setdefault("n_jobs", int(getattr(C, "RSF_NJOBS", 1)))

    log(f"[GRID] model={model_name} best_param_id={best_pid} best_params={best_params}")
    return best_params, df_grid


# -------------------------
# CV permutation importance
# -------------------------
def cv_permutation_importance(
    outcome: str,
    model_name: str,
    X_raw: pd.DataFrame,
    y_struct,
    raw_features: List[str],
    best_params: dict,
    outdir: Path,
    setting: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(outdir / "cv")
    kf = KFold(n_splits=C.N_SPLITS_CV, shuffle=True, random_state=C.SEED)

    fold_rows: List[dict] = []
    pi_rows: List[pd.DataFrame] = []

    for fold, (i_tr, i_va) in enumerate(kf.split(X_raw), start=1):
        log(f"[CV] outcome={outcome} model={model_name} setting={setting} fold={fold}/{C.N_SPLITS_CV}")

        Xtr_raw = X_raw.iloc[i_tr].copy()
        Xva_raw = X_raw.iloc[i_va].copy()
        ytr = y_struct[i_tr]
        yva = y_struct[i_va]

        pre = build_preprocessor(
            Xtr_raw,
            binary_cols=C.BINARY_COLS,
            cat_cols=C.CAT_COLS,
            ordinal_cols=C.ORDINAL_COLS,
            numeric_cols=C.NUM_COLS,
            geno_cols=C.GENO_COLS,
            low_missing_thresh=LOW_MISS,
            high_missing_thresh=HIGH_MISS,
            skew_thresh=SKEW_THRESH,
        )

        Xt = np.asarray(pre.fit_transform(Xtr_raw), dtype=np.float32)
        Xv = np.asarray(pre.transform(Xva_raw), dtype=np.float32)

        m = make_model(model_name, params=best_params)
        m.fit(Xt, ytr)

        rv = m.predict_risk(Xv)
        fold_c = cindex(yva, rv)
        aucs = auc_td_train_test(ytr, yva, rv, C.HORIZONS_YEARS)

        fold_rows.append({"outcome": outcome, "model": model_name, "setting": setting, "fold": fold, "cindex": fold_c, **aucs})

        log(f"[CV-PI] {model_name} fold={fold} repeats={C.PI_REPEATS_CV} features={len(raw_features)}")
        pi_long = permutation_importance_raw(
            model_wrap=m,
            preprocessor=pre,
            X_val_raw=Xva_raw,
            y_val_struct=yva,
            feature_list=raw_features,
            repeats=C.PI_REPEATS_CV,
            seed=C.SEED + fold,
        )
        if not pi_long.empty:
            pi_long["outcome"] = outcome
            pi_long["model"] = model_name
            pi_long["setting"] = setting
            pi_long["fold"] = fold
            pi_rows.append(pi_long)

        del m, pre, Xt, Xv, Xtr_raw, Xva_raw
        gc.collect()

    df_cv = pd.DataFrame(fold_rows)
    df_cv.to_csv(outdir / "cv" / f"cv_metrics_{setting}_{model_name}.csv", index=False)

    pi_long_all = pd.concat(pi_rows, ignore_index=True) if pi_rows else pd.DataFrame(
        columns=["feature", "repeat", "delta_cindex", "outcome", "model", "setting", "fold"]
    )
    pi_long_all.to_csv(outdir / "cv" / f"cv_perm_importance_long_{setting}_{model_name}.csv", index=False)

    pi_mean = mean_pi_table(pi_long_all)
    pi_mean.to_csv(outdir / "cv" / f"cv_perm_importance_mean_{setting}_{model_name}.csv", index=False)

    row = {
        "Outcome": outcome,
        "Models": model_name,
        "Setting": setting,
        "Mean C-index": float(df_cv["cindex"].mean()) if len(df_cv) else float("nan"),
        "SD C-index": float(df_cv["cindex"].std(ddof=1)) if len(df_cv) > 1 else float("nan"),
    }
    for h in C.HORIZONS_YEARS:
        k = f"AUC@{int(h)}yr"
        row[k] = float(df_cv[k].mean()) if k in df_cv else float("nan")
    row["Mean Permutation importance"] = float(pi_mean["mean_delta_cindex"].mean()) if len(pi_mean) else float("nan")

    df_summary = pd.DataFrame([row])
    df_summary.to_csv(outdir / "cv" / f"cv_summary_{setting}_{model_name}.csv", index=False)
    return df_cv, pi_mean, df_summary


# -------------------------
# Reduced features
# -------------------------
def reduced_tags_in_order() -> List[str]:
    # sequential order you intended
    return [f"top{int(frac * 100)}" for frac in C.REDUCED_FRACS]


def select_reduced_features(pi_mean: pd.DataFrame, raw_features: List[str]) -> Dict[str, List[str]]:
    tags = reduced_tags_in_order()
    if pi_mean.empty:
        return {t: raw_features[:] for t in tags}

    ranked = pi_mean.sort_values("mean_delta_cindex", ascending=False)["feature"].tolist()
    n = len(ranked)

    out: Dict[str, List[str]] = {}
    for frac in C.REDUCED_FRACS:
        k = max(1, int(math.ceil(frac * n)))
        out[f"top{int(frac * 100)}"] = ranked[:k]
    return out


def load_reduced_features(out_base: Path, model_name: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for tag in reduced_tags_in_order():
        p = out_base / "cv" / "reduced_lists" / f"{model_name}_{tag}.json"
        if not p.exists():
            raise RuntimeError(f"Missing reduced list {p}. You must run FULL once to create it.")
        obj = json.loads(p.read_text())
        feats = obj.get("features", None)
        if not isinstance(feats, list) or len(feats) == 0:
            raise RuntimeError(f"Invalid reduced list file {p}")
        out[tag] = feats
    return out


# -------------------------
# Final fit/save
# -------------------------
def fit_and_save_final(
    outcome: str,
    model_name: str,
    setting: str,
    df_train: pd.DataFrame,
    time_col: str,
    event_col: str,
    raw_features: List[str],
    best_params: dict,
    outdir_model: Path,
):
    log(f"[FINAL] outcome={outcome} model={model_name} setting={setting} fit on full TRAIN n={len(df_train)}")

    y_struct, mask = make_y_struct(df_train, time_col, event_col)
    X_raw = df_train.loc[mask, raw_features].copy().reset_index(drop=True)

    pre = build_preprocessor(
        X_raw,
        binary_cols=C.BINARY_COLS,
        cat_cols=C.CAT_COLS,
        ordinal_cols=C.ORDINAL_COLS,
        numeric_cols=C.NUM_COLS,
        geno_cols=C.GENO_COLS,
        low_missing_thresh=LOW_MISS,
        high_missing_thresh=HIGH_MISS,
        skew_thresh=SKEW_THRESH,
    )
    Xt = np.asarray(pre.fit_transform(X_raw), dtype=np.float32)

    m = make_model(model_name, params=best_params)
    m.fit(Xt, y_struct)

    manifest = {
        "outcome": outcome,
        "model": model_name,
        "setting": setting,
        "best_params": best_params,
        "raw_features": raw_features,
        "preprocess": {
            "low_missing_thresh": LOW_MISS,
            "high_missing_thresh": HIGH_MISS,
            "skew_thresh": SKEW_THRESH,
        },
        "seed": C.SEED,
    }
    save_model_bundle(outdir_model, m, pre, raw_features, manifest)
    log(f"[FINAL] saved bundle -> {outdir_model}")
    return m, pre


# -------------------------
# Test evaluation
# -------------------------
def evaluate_on_test(
    outcome: str,
    model_name: str,
    setting: str,
    m,
    pre,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    time_col: str,
    event_col: str,
    raw_features: List[str],
    out_base: Path,
    pi_ranked_features: List[str],
):
    ensure_dir(out_base / "test")
    ensure_dir(out_base / "plots")

    y_train, _ = make_y_struct(df_train, time_col, event_col)
    y_test, mask_test = make_y_struct(df_test, time_col, event_col)
    X_test_raw = df_test.loc[mask_test, raw_features].copy().reset_index(drop=True)

    Xt_test = np.asarray(pre.transform(X_test_raw), dtype=np.float32)
    risk_rank = m.predict_risk(Xt_test).astype(float)

    risk3, mode3 = predicted_risk_at_horizon(m, pre, X_test_raw, 3.0)
    risk5, mode5 = predicted_risk_at_horizon(m, pre, X_test_raw, 5.0)
    risk10, mode10 = predicted_risk_at_horizon(m, pre, X_test_raw, 10.0)

    S5 = survival_prob_at_t(m, Xt_test, 5.0)
    smode5 = "available" if S5 is not None else "unavailable"

    log(f"[TEST] {outcome} {model_name} {setting} risk modes: 3y={mode3} 5y={mode5} 10y={mode10} | S5_mode={smode5}")

    pred_df = pd.DataFrame({
        "time": y_test["time"].astype(float),
        "event": y_test["event"].astype(int),
        "risk_rank": risk_rank,
        "risk_3y": np.asarray(risk3, float),
        "risk_5y": np.asarray(risk5, float),
        "risk_10y": np.asarray(risk10, float),
    })
    pred_df.to_csv(out_base / "test" / f"test_predictions_{setting}_{model_name}.csv", index=False)

    point_c = cindex(y_test, risk_rank)
    boot = bootstrap_cindex(y_test, risk_rank, n_boot=C.BOOTSTRAPS, seed=C.SEED)
    aucs = auc_td_train_test(y_train, y_test, risk_rank, C.HORIZONS_YEARS)

    b5_cc = brier_at_horizon_complete_case(y_test, risk5, 5.0)

    b5_ipcw = float("nan")
    if S5 is not None:
        S5 = np.asarray(S5, dtype=float)
        if np.isfinite(S5).all():
            try:
                b5_ipcw = brier_ipcw_at_horizon(y_train, y_test, S5, 5.0)
            except Exception as e:
                log(f"[Brier-IPCW] failed -> NaN. error={e}")

    slope, intercept = (float("nan"), float("nan"))
    if mode5 in ("risk_at_horizon", "survival_fn"):
        try:
            slope, intercept = calibrate_logistic(risk5, y_test, 5.0)
        except Exception as e:
            log(f"[CAL] skipped calibration: {e}")

    row = {
        "Outcome": outcome,
        "Models": model_name,
        "Setting": setting,
        "Point C-index": float(point_c),
        **boot,
        **aucs,
        "Brier_completecase@5yr": float(b5_cc),
        "Brier_IPCW@5yr": float(b5_ipcw),
        "calibration_slope@5yr": float(slope),
        "calibration_intercept@5yr": float(intercept),
        "risk_mode@5yr": mode5,
        "surv_mode@5yr": smode5,
    }
    df_metrics = pd.DataFrame([row])
    df_metrics.to_csv(out_base / "test" / f"test_metrics_{setting}_{model_name}.csv", index=False)

    # test PI only on top-K
    topk = [f for f in pi_ranked_features if f in raw_features][:C.PI_TOPK_TEST]
    log(f"[TEST-PI] {outcome} {model_name} {setting} topk={len(topk)} repeats={C.PI_REPEATS_TEST}")

    pi_long = permutation_importance_raw(
        model_wrap=m,
        preprocessor=pre,
        X_val_raw=X_test_raw,
        y_val_struct=y_test,
        feature_list=topk,
        repeats=C.PI_REPEATS_TEST,
        seed=C.SEED,
    )
    if not pi_long.empty:
        pi_long.to_csv(out_base / "test" / f"test_perm_importance_{setting}_{model_name}.csv", index=False)
        mean_pi_table(pi_long).to_csv(out_base / "test" / f"test_perm_importance_mean_{setting}_{model_name}.csv", index=False)

    # ROC JSONs
    for h, rprob in [(3.0, risk3), (5.0, risk5), (10.0, risk10)]:
        roc = roc_at_horizon(y_test, rprob, h)
        save_json(
            {"horizon_years": h, "auc": roc["auc"], "fpr": roc["fpr"].tolist(), "tpr": roc["tpr"].tolist()},
            out_base / "plots" / f"roc_{setting}_{model_name}_{int(h)}yr.json"
        )

    # KM @ 5y
    try:
        km_info = km_plot_low_high(
            pred_df,
            time_col="time",
            event_col="event",
            risk_col="risk_5y",
            out_png=out_base / "plots" / f"km_{setting}_{model_name}_5yr.png",
            title=f"{outcome} | {model_name} | {setting} | Risk groups by median predicted 5y risk",
        )
        save_json(km_info, out_base / "plots" / f"km_{setting}_{model_name}_5yr.json")
    except Exception as e:
        log(f"[KM] skipped (lifelines missing or error): {e}")

    return df_metrics


# -------------------------
# Run a single (outcome, model) for a single setting
# -------------------------
def run_one_setting(
    outcome: str,
    model_name: str,
    setting: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    time_col: str,
    event_col: str,
    mask_train: np.ndarray,
    y_train,
    X_train_full: pd.DataFrame,
    feats: List[str],
    out_base: Path,
    seed: int,
    *,
    do_feature_ranking: bool,
) -> None:
    if is_done(outcome, model_name, setting):
        log(f"[SKIP] already done -> outcome={outcome} model={model_name} setting={setting}")
        return

    # grid search is always done per setting (because features change)
    log(f"\n----- MODEL {model_name} ({setting}) -----")
    X_train = X_train_full if setting == "full" else df_train.loc[mask_train, feats].copy().reset_index(drop=True)

    best_params, df_grid = cv_grid_search(model_name, X_train, y_train, seed=seed)
    df_grid.to_csv(out_base / "cv" / f"grid_metrics_{setting}_{model_name}.csv", index=False)

    # PI + reduced list generation only on FULL
    pi_mean = pd.DataFrame()
    if do_feature_ranking:
        _, pi_mean, df_sum = cv_permutation_importance(
            outcome=outcome,
            model_name=model_name,
            X_raw=X_train,
            y_struct=y_train,
            raw_features=feats,
            best_params=best_params,
            outdir=out_base,
            setting=setting,
        )
        df_sum.to_csv(out_base / "cv" / f"cv_summary_{setting}_{model_name}.csv", index=False)

        if setting == "full":
            reduced = select_reduced_features(pi_mean, feats)
            ensure_dir(out_base / "cv" / "reduced_lists")
            for tag, fts in reduced.items():
                save_json({"features": fts}, out_base / "cv" / "reduced_lists" / f"{model_name}_{tag}.json")

    # final fit/save
    model_dir = ensure_dir(out_base / "final_models" / setting / model_name)
    m, pre = fit_and_save_final(
        outcome, model_name, setting,
        df_train, time_col, event_col,
        feats, best_params, model_dir
    )

    # ranking features for test-PI (if no pi_mean, just use feats)
    ranked_feats = (
        pi_mean.sort_values("mean_delta_cindex", ascending=False)["feature"].tolist()
        if len(pi_mean) else feats
    )

    tm = evaluate_on_test(
        outcome, model_name, setting,
        m, pre,
        df_train, df_test,
        time_col, event_col,
        feats, out_base,
        pi_ranked_features=ranked_feats,
    )
    tm.to_csv(out_base / "test" / f"test_metrics_{setting}_{model_name}.csv", index=False)

    del m, pre, X_train
    gc.collect()


# -------------------------
# Run one outcome end-to-end with resume controls
# -------------------------
def run_outcome(
    outcome: str,
    *,
    start_model: Optional[str] = None,
    start_setting: str = "full",
    skip_done: bool = True,
) -> None:
    spec = C.OUTCOMES[outcome]
    time_col, event_col = spec["time_col"], spec["event_col"]

    out_base = ensure_dir(outcome_dir(outcome))
    ensure_dir(out_base / "cv")
    ensure_dir(out_base / "test")
    ensure_dir(out_base / "plots")
    ensure_dir(out_base / "final_models")
    ensure_dir(out_base / "splits_used")

    log(f"\n========== RUN outcome={outcome} ==========")
    log(f"[LOAD] reading {C.DATA_PATH}")
    df = pd.read_csv(C.DATA_PATH)

    df_train, df_test, train_idx, test_idx = make_train_test(df, outcome)

    pd.Series(train_idx).to_csv(out_base / "splits_used" / "train_idx.csv", index=False, header=False)
    pd.Series(test_idx).to_csv(out_base / "splits_used" / "test_idx.csv", index=False, header=False)

    df_train = clean_for_outcome(df_train, time_col, event_col)
    df_test = clean_for_outcome(df_test, time_col, event_col)

    log(f"[INFO] train events={int(df_train[event_col].sum())} / n={len(df_train)}")
    log(f"[INFO] test  events={int(df_test[event_col].sum())} / n={len(df_test)}")

    y_train, mask_train = make_y_struct(df_train, time_col, event_col)
    X_train_full = df_train.loc[mask_train, C.RAW_FEATURES].copy().reset_index(drop=True)

    # model ordering with start control
    model_list = list(C.MODEL_NAMES)
    if start_model is not None:
        if start_model not in model_list:
            raise ValueError(f"start_model={start_model} not in MODEL_NAMES={model_list}")
        model_list = model_list[model_list.index(start_model):]

    # setting ordering with start control
    red_tags = reduced_tags_in_order()
    setting_order = ["full"] + red_tags
    if start_setting not in setting_order:
        raise ValueError(f"start_setting={start_setting} must be one of {setting_order}")
    start_setting_idx = setting_order.index(start_setting)

    for model_name in model_list:
        # If we start at a reduced setting, we must load reduced lists created earlier.
        reduced = {}
        if start_setting_idx > 0:
            reduced = load_reduced_features(out_base, model_name)

        # FULL (if allowed by start_setting)
        if start_setting_idx == 0:
            run_one_setting(
                outcome, model_name, "full",
                df_train, df_test, time_col, event_col,
                mask_train, y_train, X_train_full,
                feats=C.RAW_FEATURES,
                out_base=out_base,
                seed=C.SEED,
                do_feature_ranking=True,
            )
            reduced = load_reduced_features(out_base, model_name)

        # Reduced sequential
        for tag in red_tags[start_setting_idx - 1 if start_setting_idx > 0 else 0:]:
            feats = reduced[tag]
            run_one_setting(
                outcome, model_name, tag,
                df_train, df_test, time_col, event_col,
                mask_train, y_train, X_train_full,
                feats=feats,
                out_base=out_base,
                seed=C.SEED + 17,
                do_feature_ranking=True,
            )

        # after first model, always revert start_setting to full for subsequent models
        start_setting_idx = 0

    # posthoc logrank tables (once)
    try:
        lr_train = logrank_feature_table(df_train, time_col, event_col, C.RAW_FEATURES)
        lr_train.to_csv(out_base / "test" / "logrank_features_train.csv", index=False)

        lr_test = logrank_feature_table(df_test, time_col, event_col, C.RAW_FEATURES)
        lr_test.to_csv(out_base / "test" / "logrank_features_test.csv", index=False)
    except Exception as e:
        log(f"[POSTHOC] logrank feature table skipped: {e}")

    log(f"========== DONE outcome={outcome} ==========\n")


# -------------------------
# Run multiple outcomes sequentially with resume controls
# -------------------------
def run_all(
    *,
    outcomes_order: Optional[List[str]] = None,
    start_outcome: Optional[str] = None,
    start_model: Optional[str] = None,
    start_setting: str = "full",
) -> None:
    """
    This is what you want for your restart:

    - outcomes_order defaults to config order
    - start_outcome/start_model/start_setting apply ONLY to the first outcome,
      then all remaining outcomes run from scratch (full then reduced).
    """
    outcomes = outcomes_order or list(C.OUTCOMES.keys())
    if start_outcome is not None:
        if start_outcome not in outcomes:
            raise ValueError(f"start_outcome={start_outcome} not in outcomes={outcomes}")
        outcomes = outcomes[outcomes.index(start_outcome):]

    for i, oc in enumerate(outcomes):
        if i == 0:
            run_outcome(oc, start_model=start_model, start_setting=start_setting)
        else:
            run_outcome(oc, start_model=None, start_setting="full")
