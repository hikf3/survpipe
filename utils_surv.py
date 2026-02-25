# surv2/utils_surv.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump

# scikit-survival
try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
    from sksurv.metrics import brier_score as sk_brier_score
except Exception as e:
    Surv = None
    concordance_index_censored = None
    cumulative_dynamic_auc = None
    sk_brier_score = None
    _SKSURV_METRICS_ERROR = e
else:
    _SKSURV_METRICS_ERROR = None

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression

# lifelines (KM + logrank + at-risk table)
try:
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import logrank_test, multivariate_logrank_test
except Exception:
    KaplanMeierFitter = None
    add_at_risk_counts = None
    logrank_test = None
    multivariate_logrank_test = None


# -----------------------
# Logging
# -----------------------
def log(msg: str):
    print(msg, flush=True)


# -----------------------
# FS helpers
# -----------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))


def save_txt(lines: List[str], path: Path):
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n")


# -----------------------
# Survival helpers
# -----------------------
def require_sksurv_metrics():
    if _SKSURV_METRICS_ERROR is not None:
        raise ImportError(
            "scikit-survival metrics are required. "
            f"Import error: {_SKSURV_METRICS_ERROR}"
        )


def make_y_struct(df: pd.DataFrame, time_col: str, event_col: str):
    """
    Returns:
      y_struct: Surv array
      mask: boolean mask of rows kept (valid time/event)
    """
    require_sksurv_metrics()
    t = pd.to_numeric(df[time_col], errors="coerce").astype(float).to_numpy()
    e = pd.to_numeric(df[event_col], errors="coerce").astype(float).to_numpy()

    mask = np.isfinite(t) & np.isfinite(e) & (t > 0)
    t = t[mask]
    e = e[mask]
    e = (e > 0).astype(bool)

    return Surv.from_arrays(event=e, time=t), mask


def cindex(y_struct, risk: np.ndarray) -> float:
    require_sksurv_metrics()
    ev = y_struct["event"]
    tt = y_struct["time"]
    return float(concordance_index_censored(ev, tt, np.asarray(risk, dtype=float))[0])


def auc_td_train_test(y_train, y_test, risk_test, horizons_years):
    """
    Time-dependent AUC at specific horizons.
    Uses cumulative_dynamic_auc which requires y_train.
    """
    require_sksurv_metrics()
    out = {}
    for t in horizons_years:
        try:
            aucs, _ = cumulative_dynamic_auc(
                y_train, y_test,
                np.asarray(risk_test, float),
                np.asarray([float(t)], float),
            )
            out[f"AUC@{int(t)}yr"] = float(aucs[0])
        except Exception:
            out[f"AUC@{int(t)}yr"] = float("nan")
    return out


def horizon_labels(y_struct, horizon_years: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build y_binary and known mask:
      case: event==1 and time <= h
      non-case: time > h
      unknown: censored before h -> excluded
    """
    t = y_struct["time"].astype(float)
    e = y_struct["event"].astype(bool)
    h = float(horizon_years)

    case = e & (t <= h)
    noncase = (t > h)
    known = case | noncase
    yb = np.zeros_like(t, dtype=int)
    yb[case] = 1
    return yb, known


def brier_at_horizon_complete_case(y_struct, pred_prob: np.ndarray, horizon_years: float) -> float:
    """
    Complete-case horizon Brier:
      - drops censored-before-horizon
      - NOT IPCW
    """
    yb, known = horizon_labels(y_struct, horizon_years)
    yb = yb[known]
    pp = np.asarray(pred_prob, dtype=float)[known]
    return float(np.mean((pp - yb) ** 2))


def calibrate_logistic(pred_prob: np.ndarray, y_struct, horizon_years: float) -> Tuple[float, float]:
    """
    Logistic calibration: y ~ intercept + slope * logit(p)
    Returns (slope, intercept)
    """
    yb, known = horizon_labels(y_struct, horizon_years)
    yb = yb[known]
    p = np.clip(np.asarray(pred_prob, dtype=float)[known], 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)

    # penalty="none" works on sklearn>=1.2; if your env complains, use penalty=None
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x, yb)
    slope = float(lr.coef_.reshape(-1)[0])
    intercept = float(lr.intercept_.reshape(-1)[0])
    return slope, intercept


def bootstrap_cindex(y_struct, risk: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(risk)
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals.append(cindex(y_struct[idx], np.asarray(risk, dtype=float)[idx]))
    vals = np.asarray(vals, dtype=float)
    return {
        "bootstrap_mean_cindex": float(vals.mean()),
        "bootstrap_se_cindex": float(vals.std(ddof=1)),
        "bootstrap_ci2p5": float(np.quantile(vals, 0.025)),
        "bootstrap_ci97p5": float(np.quantile(vals, 0.975)),
    }


def roc_at_horizon(y_struct, score: np.ndarray, horizon_years: float) -> Dict[str, Any]:
    yb, known = horizon_labels(y_struct, horizon_years)
    yb = yb[known]
    sc = np.asarray(score, dtype=float)[known]
    fpr, tpr, thr = roc_curve(yb, sc)
    return {
        "fpr": fpr, "tpr": tpr, "thresholds": thr,
        "auc": float(auc(fpr, tpr)),
        "n_known": int(known.sum()),
        "n_case": int(yb.sum()),
    }


# -----------------------
# Survival probability at a horizon (needed for IPCW Brier)
# -----------------------
def survival_prob_at_t(model_wrap, X_transformed: np.ndarray, t_years: float) -> Optional[np.ndarray]:
    """
    Returns S(t|x) per sample if survival function exists, else None.
    """
    sf = model_wrap.predict_survival_fn(X_transformed)
    if sf is None:
        return None

    t = float(t_years)
    out = []
    for s in sf:
        if hasattr(s, "__call__"):
            out.append(float(s(t)))
        elif isinstance(s, tuple) and len(s) == 2:
            times, surv = s
            times = np.asarray(times, dtype=float)
            surv = np.asarray(surv, dtype=float)
            idx = np.searchsorted(times, t, side="right") - 1
            idx = int(np.clip(idx, 0, len(times) - 1))
            out.append(float(surv[idx]))
        else:
            return None
    return np.asarray(out, dtype=float)


def brier_ipcw_at_horizon(y_train, y_test, surv_prob_test: np.ndarray, horizon_years: float) -> float:
    """
    IPCW Brier score at a single horizon using sksurv.metrics.brier_score.

    Inputs:
      surv_prob_test: S(t|x) for each test sample at the horizon time.
    """
    require_sksurv_metrics()
    if sk_brier_score is None:
        raise ImportError("sksurv.metrics.brier_score is not available in this environment.")

    times = np.asarray([float(horizon_years)], dtype=float)
    surv_prob_test = np.asarray(surv_prob_test, dtype=float).reshape(-1, 1)

    _, scores = sk_brier_score(y_train, y_test, surv_prob_test, times)
    return float(scores[0])


# -----------------------
# Predicted event probability by horizon (what your pipeline wants)
# -----------------------
def predicted_risk_at_horizon(model_wrap, preprocessor, X_raw: pd.DataFrame, t_years: float) -> Tuple[np.ndarray, str]:
    """
    Returns:
      prob: P(event by t_years)
      mode: description of method used
    """
    Xt = np.asarray(preprocessor.transform(X_raw), dtype=np.float32)

    prob = model_wrap.predict_risk_at(Xt, t_years)
    if prob is not None:
        return np.asarray(prob, dtype=float), "risk_at_horizon"

    # Fallback if survival exists but wrapper didn't expose risk_at
    sf = model_wrap.predict_survival_fn(Xt)
    if sf is not None:
        out = []
        t = float(t_years)
        for s in sf:
            if hasattr(s, "__call__"):
                out.append(float(1.0 - s(t)))
            elif isinstance(s, tuple) and len(s) == 2:
                times, surv = s
                times = np.asarray(times, dtype=float)
                surv = np.asarray(surv, dtype=float)
                idx = np.searchsorted(times, t, side="right") - 1
                idx = int(np.clip(idx, 0, len(times) - 1))
                out.append(float(1.0 - surv[idx]))
            else:
                break
        if len(out) == len(X_raw):
            return np.asarray(out, dtype=float), "survival_fn"

    # Last resort: min-max scaled risk score (ranking only)
    risk = model_wrap.predict_risk(Xt).astype(float)
    r01 = (risk - np.nanmin(risk)) / (np.nanmax(risk) - np.nanmin(risk) + 1e-12)
    return r01, "scaled_rank_risk"


# -----------------------
# Permutation importance (raw features)
# -----------------------
def permutation_importance_raw(
    model_wrap,
    preprocessor,
    X_val_raw: pd.DataFrame,
    y_val_struct,
    feature_list: List[str],
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Permute RAW features one at a time on validation set, transform + score C-index.
    Returns long df with: feature, repeat, delta_cindex
    """
    rng = np.random.default_rng(seed)

    X0 = np.asarray(preprocessor.transform(X_val_raw), dtype=np.float32)
    risk0 = model_wrap.predict_risk(X0)
    base = cindex(y_val_struct, risk0)

    rows = []
    for f in feature_list:
        if f not in X_val_raw.columns:
            continue
        col = X_val_raw[f].to_numpy(copy=True)
        for r in range(int(repeats)):
            Xp = X_val_raw.copy()
            perm = col.copy()
            rng.shuffle(perm)
            Xp[f] = perm
            Xp_t = np.asarray(preprocessor.transform(Xp), dtype=np.float32)
            riskp = model_wrap.predict_risk(Xp_t)
            cp = cindex(y_val_struct, riskp)
            rows.append({"feature": f, "repeat": r, "delta_cindex": float(base - cp)})
    return pd.DataFrame(rows)


def mean_pi_table(pi_long: pd.DataFrame) -> pd.DataFrame:
    if pi_long.empty:
        return pd.DataFrame(columns=["feature", "mean_delta_cindex"])
    g = pi_long.groupby("feature")["delta_cindex"].mean().sort_values(ascending=False)
    return g.reset_index().rename(columns={"delta_cindex": "mean_delta_cindex"})


# -----------------------
# Model saving bundle
# -----------------------
def save_model_bundle(outdir: Path, model_wrap, preprocessor, raw_features: List[str], manifest: dict):
    ensure_dir(outdir)
    dump(model_wrap, outdir / "model.joblib")
    dump(preprocessor, outdir / "preprocessor.joblib")
    save_json({"raw_features": raw_features}, outdir / "features_raw.json")
    save_json(manifest, outdir / "training_manifest.json")


# -----------------------
# Log-rank feature tests (model-agnostic)
# -----------------------
def require_lifelines():
    if KaplanMeierFitter is None or logrank_test is None:
        raise ImportError("lifelines is required for KM/log-rank plots and feature log-rank tests.")


def logrank_feature_table(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    feature_list: List[str],
    *,
    max_levels: int = 8,
) -> pd.DataFrame:
    """
    For each feature:
      - binary: groups 0 vs 1
      - numeric: median split
      - categorical: global multivariate log-rank across levels (up to max_levels), else collapse rare into 'Other'
    """
    require_lifelines()

    x = df.copy()
    x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    x[event_col] = pd.to_numeric(x[event_col], errors="coerce").astype(int)
    x = x[np.isfinite(x[time_col].to_numpy()) & np.isfinite(x[event_col].to_numpy()) & (x[time_col] > 0)]
    x[event_col] = (x[event_col] > 0).astype(int)

    rows = []
    for f in feature_list:
        if f not in x.columns:
            continue

        s = x[f]
        s_num = pd.to_numeric(s, errors="coerce")
        is_numeric = (s_num.notna().mean() > 0.8)

        # binary numeric-coded
        if s.dropna().nunique() <= 2 and not (s.dtype == object):
            g = s.fillna(0).astype(int).values
            mask0 = (g == 0)
            mask1 = (g == 1)
            if mask0.sum() < 10 or mask1.sum() < 10:
                pval = np.nan
            else:
                res = logrank_test(
                    x.loc[mask0, time_col], x.loc[mask1, time_col],
                    x.loc[mask0, event_col], x.loc[mask1, event_col]
                )
                pval = float(res.p_value)

            rows.append({
                "feature": f,
                "test": "logrank_2group",
                "p_value": pval,
                "n0": int(mask0.sum()),
                "n1": int(mask1.sum()),
                "events0": int(x.loc[mask0, event_col].sum()),
                "events1": int(x.loc[mask1, event_col].sum()),
            })
            continue

        if is_numeric:
            med = float(np.nanmedian(s_num.values))
            grp = np.where(s_num.values <= med, "Low", "High")
            maskL = (grp == "Low")
            maskH = (grp == "High")
            if maskL.sum() < 10 or maskH.sum() < 10:
                pval = np.nan
            else:
                res = logrank_test(
                    x.loc[maskL, time_col], x.loc[maskH, time_col],
                    x.loc[maskL, event_col], x.loc[maskH, event_col]
                )
                pval = float(res.p_value)

            rows.append({
                "feature": f,
                "test": "logrank_median_split",
                "p_value": pval,
                "median": med,
                "n_low": int(maskL.sum()),
                "n_high": int(maskH.sum()),
                "events_low": int(x.loc[maskL, event_col].sum()),
                "events_high": int(x.loc[maskH, event_col].sum()),
            })
            continue

        # categorical
        ss = s.fillna("__MISSING__").astype(str)
        vc = ss.value_counts(dropna=False)

        if len(vc) > max_levels:
            keep = vc.index[:max_levels - 1].tolist()
            ss = ss.where(ss.isin(keep), other="Other")

        try:
            res = multivariate_logrank_test(
                x[time_col].values,
                groups=ss.values,
                event_observed=x[event_col].values
            )
            pval = float(res.p_value)
        except Exception:
            pval = np.nan

        grp_counts = ss.value_counts()
        ev_counts = x.groupby(ss)[event_col].sum()

        rows.append({
            "feature": f,
            "test": "logrank_multigroup",
            "p_value": pval,
            "groups": "|".join(grp_counts.index.tolist()),
            "group_sizes": "|".join([str(int(v)) for v in grp_counts.values]),
            "group_events": "|".join([str(int(ev_counts.loc[g])) for g in grp_counts.index]),
        })

    return pd.DataFrame(rows)


# -----------------------
# “Deviance-like” nested comparison (score-based; works for any model)
# -----------------------
def performance_delta_bootstrap(
    y_struct,
    metric_fn,
    score_full: np.ndarray,
    score_reduced: np.ndarray,
    n_boot: int = 500,
    seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(score_full)
    deltas = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        mf = metric_fn(y_struct[idx], score_full[idx])
        mr = metric_fn(y_struct[idx], score_reduced[idx])
        deltas.append(mf - mr)
    deltas = np.asarray(deltas, dtype=float)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_se": float(deltas.std(ddof=1)),
        "delta_ci2p5": float(np.quantile(deltas, 0.025)),
        "delta_ci97p5": float(np.quantile(deltas, 0.975)),
    }


# -----------------------
# KM plot with CI + logrank + at-risk table
# -----------------------
def km_plot_low_high(
    df_pred: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    risk_col: str,
    out_png: Path,
    title: str,
):
    """
    Median split -> Low/High risk groups with:
      - KM curves + 95% CI
      - log-rank p-value
      - number-at-risk table
      - save PNG
    """
    require_lifelines()
    import matplotlib.pyplot as plt

    x = df_pred.copy()
    x[time_col] = pd.to_numeric(x[time_col], errors="coerce")
    x[event_col] = pd.to_numeric(x[event_col], errors="coerce")
    x[risk_col] = pd.to_numeric(x[risk_col], errors="coerce")

    x = x[np.isfinite(x[time_col].to_numpy()) & np.isfinite(x[event_col].to_numpy()) & np.isfinite(x[risk_col].to_numpy())]
    x = x[x[time_col] > 0]
    x[event_col] = (x[event_col].astype(int) > 0).astype(int)

    med = float(np.nanmedian(x[risk_col].values))
    x["risk_group"] = np.where(x[risk_col].values <= med, "Low", "High")

    low = x["risk_group"] == "Low"
    high = x["risk_group"] == "High"
    pval = np.nan
    if low.sum() >= 5 and high.sum() >= 5:
        res = logrank_test(
            x.loc[low, time_col], x.loc[high, time_col],
            x.loc[low, event_col], x.loc[high, event_col]
        )
        pval = float(res.p_value)

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    fig = plt.figure()
    ax = plt.gca()

    kmf_low.fit(x.loc[low, time_col], event_observed=x.loc[low, event_col], label=f"Low (n={low.sum()})")
    kmf_high.fit(x.loc[high, time_col], event_observed=x.loc[high, event_col], label=f"High (n={high.sum()})")

    kmf_low.plot_survival_function(ax=ax, ci_show=True)
    kmf_high.plot_survival_function(ax=ax, ci_show=True)

    ax.set_title(f"{title}\nMedian split; log-rank p={pval:.3g}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival probability")

    if add_at_risk_counts is not None:
        add_at_risk_counts(kmf_low, kmf_high, ax=ax)

    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"median_risk": med, "logrank_p": pval, "n_low": int(low.sum()), "n_high": int(high.sum())}
