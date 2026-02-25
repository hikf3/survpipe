# surv2/preprocess.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator


# -----------------------------------------------------------------------------
# Rationale (manuscript-ready paragraph)
# -----------------------------------------------------------------------------
IMPUTATION_RATIONALE = """
We used a simple, leakage-safe, and defensible missingness policy for EHR tabular data.
All preprocessing was fit on the training split only. For numerical/genotype features,
we applied median imputation, and added explicit missingness indicators for features
with non-trivial missingness. We did not use KNN imputation or MICE because (i) with
~3,800 training patients and mixed feature types, distance-based neighbors can be
unstable and sensitive to scaling and correlation structure, (ii) KNN/MICE can
implicitly borrow information across patients in ways that complicate reproducibility
and increase compute cost in repeated cross-validation, and (iii) in EHR data, the
pattern of missingness is often informative (MNAR); preserving this signal via
missingness indicators is typically more robust than attempting to “guess” values via
multivariate imputation.
""".strip()


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Coerce input to float numpy array; invalid parses -> NaN."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            arr = X.apply(pd.to_numeric, errors="coerce").to_numpy()
        else:
            arr = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()
        return arr.astype(float)


class SkewLogRMSScaler(BaseEstimator, TransformerMixin):
    """
    Optional stabilizer:
      - If abs(skewness) > skew_thresh: signed log1p transform
      - RMS scaling: x / sqrt(mean(x^2))
    """
    def __init__(self, skew_thresh: float = 3.0, eps: float = 1e-12):
        self.skew_thresh = float(skew_thresh)
        self.eps = float(eps)
        self.use_log_: np.ndarray | None = None
        self.rms_: np.ndarray | None = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        s = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[np.isfinite(col)]
            if len(col) < 3:
                s.append(0.0)
                continue
            m = float(col.mean())
            v = float(col.var()) + self.eps
            skew = float(np.mean(((col - m) / np.sqrt(v)) ** 3))
            s.append(skew)

        s = np.asarray(s, dtype=float)
        self.use_log_ = np.abs(s) > self.skew_thresh

        X2 = self._apply_log(X.copy())
        self.rms_ = np.sqrt(np.mean(X2 ** 2, axis=0) + self.eps)
        self.rms_ = np.maximum(self.rms_, self.eps)
        return self

    def _apply_log(self, X):
        if self.use_log_ is None:
            return X
        for j, flag in enumerate(self.use_log_):
            if not flag:
                continue
            col = X[:, j]
            X[:, j] = np.sign(col) * np.log1p(np.abs(col))
        return X

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = self._apply_log(X.copy())
        return X / self.rms_


class NamedColumnTransformer(ColumnTransformer):
    """ColumnTransformer with robust get_feature_names_out()."""
    def get_feature_names_out(self, input_features=None):
        names = []
        for name, trans, cols in self.transformers_:
            if trans == "drop":
                continue

            cols_list = list(cols)

            if trans == "passthrough":
                names.extend([f"{name}__{c}" for c in cols_list])
                continue

            trans_to_query = trans
            if isinstance(trans, Pipeline):
                trans_to_query = trans.steps[-1][1]

            if hasattr(trans_to_query, "get_feature_names_out"):
                try:
                    ohe_names = trans_to_query.get_feature_names_out(cols_list)
                    names.extend([f"{name}__{n}" for n in ohe_names])
                    continue
                except Exception:
                    pass

            names.extend([f"{name}__{c}" for c in cols_list])

        return np.asarray(names, dtype=object)


def _missing_rate_partition(X: pd.DataFrame, cols: list[str], low: float, high: float):
    """
    Partition columns into:
      - low_missing: miss < low
      - mid_missing: low <= miss < high
      - high_missing: miss >= high
    """
    cols = [c for c in cols if c in X.columns]
    if not cols:
        return [], [], []

    miss = X[cols].isna().mean(axis=0)
    low_cols = miss[miss < low].index.tolist()
    mid_cols = miss[(miss >= low) & (miss < high)].index.tolist()
    high_cols = miss[miss >= high].index.tolist()
    return low_cols, mid_cols, high_cols


def build_preprocessor_for(
    X_train: pd.DataFrame,
    *,
    binary_cols: list[str],
    cat_cols: list[str],
    ordinal_cols: list[str],
    numeric_cols: list[str],
    geno_cols: list[str],
    low_missing_thresh: float = 0.20,
    high_missing_thresh: float = 0.50,
    skew_thresh: float = 3.0,
):
    """
    Clean imputation policy (computed on TRAIN only):
      - categorical: constant '__MISSING__' then one-hot
      - binary/ordinal: numeric coercion then constant fill 0
      - numeric+geno:
          < low_missing_thresh: median + scaler
          >= low_missing_thresh: median + scaler + MissingIndicator(features="all")

    Notes:
      - We intentionally do not use KNNImputer or MICE.
      - MissingIndicator is added for mid/high missingness because missingness is often informative in EHR.
    """
    X_train = X_train.copy()

    bin_cols = [c for c in binary_cols if c in X_train.columns]
    ord_cols = [c for c in ordinal_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    num_all = [c for c in (list(numeric_cols) + list(geno_cols)) if c in X_train.columns]
    num_low, num_mid, num_high = _missing_rate_partition(
        X_train, num_all, float(low_missing_thresh), float(high_missing_thresh)
    )
    num_with_ind = num_mid + num_high

    bo_pipe = Pipeline(steps=[
        ("num", NumericCoercer()),
        ("imp", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    num_pipe = Pipeline(steps=[
        ("num", NumericCoercer()),
        ("imp", SimpleImputer(strategy="median")),
        ("sklog_rms", SkewLogRMSScaler(skew_thresh=float(skew_thresh))),
    ])

    ind_pipe = Pipeline(steps=[
        ("num", NumericCoercer()),
        ("ind", MissingIndicator(features="all")),
    ])

    transformers = []
    if bin_cols:
        transformers.append(("bin", bo_pipe, bin_cols))
    if ord_cols:
        transformers.append(("ord", bo_pipe, ord_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if num_low:
        transformers.append(("num_low", num_pipe, num_low))
    if num_with_ind:
        transformers.append(("num_medhi", num_pipe, num_with_ind))
        transformers.append(("num_medhi_miss", ind_pipe, num_with_ind))

    pre = NamedColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    return pre


def build_preprocessor(
    X_train: pd.DataFrame,
    binary_cols: list[str],
    cat_cols: list[str],
    ordinal_cols: list[str],
    numeric_cols: list[str],
    geno_cols: list[str],
    *,
    low_missing_thresh: float = 0.20,
    high_missing_thresh: float = 0.50,
    skew_thresh: float = 3.0,
):
    return build_preprocessor_for(
        X_train,
        binary_cols=binary_cols,
        cat_cols=cat_cols,
        ordinal_cols=ordinal_cols,
        numeric_cols=numeric_cols,
        geno_cols=geno_cols,
        low_missing_thresh=low_missing_thresh,
        high_missing_thresh=high_missing_thresh,
        skew_thresh=skew_thresh,
    )
