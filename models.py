# surv2/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# scikit-survival
try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
except Exception as e:
    CoxnetSurvivalAnalysis = None
    RandomSurvivalForest = None
    GradientBoostingSurvivalAnalysis = None
    _SKSURV_IMPORT_ERROR = e
else:
    _SKSURV_IMPORT_ERROR = None

# CoxNN (PyTorch)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


@dataclass
class ModelWrap:
    """
    Uniform wrapper so pipeline can call:
      - fit(X, y_struct)
      - predict_risk(X)                  -> risk score for ranking (C-index, tdAUC)
      - predict_survival_fn(X)           -> optional survival function per sample
      - predict_risk_at(X, t_years)      -> optional P(event by t)
    """
    name: str
    model: Any

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict_risk(self, X) -> np.ndarray:
        # scikit-survival models: predict() returns risk score
        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X), dtype=float)
        raise RuntimeError(f"{self.name} has no predict()")

    def predict_survival_fn(self, X):
        # scikit-survival models implement predict_survival_function
        if hasattr(self.model, "predict_survival_function"):
            return self.model.predict_survival_function(X)
        return None

    def predict_risk_at(self, X, t_years: float) -> Optional[np.ndarray]:
        """
        Return P(event by t) if survival curve is available:
          P(T <= t) = 1 - S(t)
        """
        # allow model-specific implementation
        if hasattr(self.model, "predict_risk_at"):
            return self.model.predict_risk_at(X, t_years)

        sf = self.predict_survival_fn(X)
        if sf is None:
            return None

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
                return None
        return np.asarray(out, dtype=float)


def _require_sksurv():
    if _SKSURV_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-survival is required for Coxnet/RSF/GBSA. "
            f"Import error: {_SKSURV_IMPORT_ERROR}"
        )


def make_model(model_name: str, params: Optional[Dict[str, Any]] = None) -> ModelWrap:
    """
    IMPORTANT FIX:
      - CoxnetSurvivalAnalysis must be constructed with fit_baseline_model=True
        if you want predict_survival_function() / risk@horizon to work.
    """
    params = params or {}

    if model_name in ("Coxnet-EN", "Coxnet-LASSO"):
        _require_sksurv()
        l1_ratio = 0.5 if model_name == "Coxnet-EN" else 1.0

        m = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alpha_min_ratio=float(params.get("alpha_min_ratio", 0.01)),
            max_iter=int(params.get("max_iter", 20000)),
            # ✅ this is the correct place (constructor, not fit())
            fit_baseline_model=True,
        )
        return ModelWrap(model_name, m)

    if model_name == "RSF":
        _require_sksurv()
        m = RandomSurvivalForest(
            n_estimators=int(params.get("n_estimators", 500)),
            min_samples_leaf=int(params.get("min_samples_leaf", 5)),
            min_samples_split=int(params.get("min_samples_split", 10)),
            max_features=params.get("max_features", "sqrt"),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=int(params.get("random_state", 0)),
        )
        return ModelWrap(model_name, m)

    if model_name == "GBSA":
        _require_sksurv()
        m = GradientBoostingSurvivalAnalysis(
            learning_rate=float(params.get("learning_rate", 0.05)),
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 2)),
            random_state=int(params.get("random_state", 0)),
        )
        return ModelWrap(model_name, m)

    if model_name == "CoxNN":
        return ModelWrap(model_name, CoxNN(params=params))

    raise ValueError(f"Unknown model_name={model_name}")


# ----------------------
# CoxNN with baseline hazard (Breslow)
# ----------------------
class CoxNN:
    """
    Cox-nnet style:
      - MLP -> linear predictor eta(x)
      - optimize negative partial log-likelihood
      - AFTER training, estimate baseline cumulative hazard H0(t) via Breslow
        on training data.
    """
    def __init__(self, params: Dict[str, Any]):
        if torch is None:
            raise ImportError("PyTorch is required for CoxNN.")
        self.params = params
        self.net = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # baseline hazard state
        self.event_times_: Optional[np.ndarray] = None
        self.cum_baseline_hazard_: Optional[np.ndarray] = None

    def _build(self, d: int):
        pdrop = float(self.params.get("pdrop", 0.10))
        hidden = max(16, d // 4)
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(hidden, 1),
        ).to(self.device)

    @staticmethod
    def _cox_ph_loss(eta, time, event):
        order = torch.argsort(time, descending=True)
        eta = eta[order]
        time = time[order]
        event = event[order]

        hazard = torch.exp(eta)
        log_cum_hazard = torch.log(torch.cumsum(hazard, dim=0) + 1e-12)

        denom = torch.sum(event) + 1e-12
        neg_ll = -torch.sum((eta - log_cum_hazard) * event) / denom
        return neg_ll

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        event = y["event"].astype(np.float32)
        time = y["time"].astype(np.float32)

        n, d = X.shape
        if self.net is None:
            self._build(d)

        lr = float(self.params.get("lr", 1e-3))
        wd = float(self.params.get("wd", 1e-4))
        epochs = int(self.params.get("epochs", 250))
        patience = int(self.params.get("patience", 25))

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)

        # internal val split for early stopping (train-only)
        rng = np.random.default_rng(0)
        idx = np.arange(n)
        rng.shuffle(idx)
        split = int(0.85 * n)
        tr_idx, va_idx = idx[:split], idx[split:]

        Xtr = torch.from_numpy(X[tr_idx]).to(self.device)
        ttr = torch.from_numpy(time[tr_idx]).to(self.device)
        etr = torch.from_numpy(event[tr_idx]).to(self.device)

        Xva = torch.from_numpy(X[va_idx]).to(self.device)
        tva = torch.from_numpy(time[va_idx]).to(self.device)
        eva = torch.from_numpy(event[va_idx]).to(self.device)

        best = float("inf")
        best_state = None
        bad = 0

        for _ep in range(epochs):
            self.net.train()
            opt.zero_grad()
            eta = self.net(Xtr).reshape(-1)
            loss = self._cox_ph_loss(eta, ttr, etr)
            loss.backward()
            opt.step()

            self.net.eval()
            with torch.no_grad():
                veta = self.net(Xva).reshape(-1)
                vloss = float(self._cox_ph_loss(veta, tva, eva).item())

            if vloss < best - 1e-5:
                best = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        # Baseline hazard estimation (Breslow) on full training used here
        self._fit_baseline_breslow(X, time, event)
        return self

    def _predict_eta_np(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X).to(self.device)
            eta = self.net(xt).reshape(-1).detach().cpu().numpy().astype(float)
        return eta

    def _fit_baseline_breslow(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        eta = self._predict_eta_np(X)
        exp_eta = np.exp(eta)

        t = np.asarray(time, dtype=float)
        e = (np.asarray(event, dtype=float) > 0.0).astype(int)

        event_times = np.unique(t[e == 1])
        event_times.sort()

        H = 0.0
        cum_h = []
        for tj in event_times:
            dj = int(np.sum((t == tj) & (e == 1)))
            risk_sum = float(np.sum(exp_eta[t >= tj]))
            if risk_sum <= 0:
                continue
            H += dj / risk_sum
            cum_h.append(H)

        self.event_times_ = event_times
        self.cum_baseline_hazard_ = np.asarray(cum_h, dtype=float)

    def predict(self, X):
        # risk score for ranking
        return self._predict_eta_np(np.asarray(X, dtype=np.float32))

    def predict_survival_function(self, X):
        if self.event_times_ is None or self.cum_baseline_hazard_ is None or len(self.cum_baseline_hazard_) == 0:
            return None

        eta = self._predict_eta_np(np.asarray(X, dtype=np.float32))
        exp_eta = np.exp(eta)

        times = self.event_times_
        H0 = self.cum_baseline_hazard_

        out = []
        for r in exp_eta:
            surv = np.exp(-H0 * r)
            out.append((times.copy(), surv.astype(float)))
        return out

    def predict_risk_at(self, X, t_years: float) -> np.ndarray:
        sf = self.predict_survival_function(X)
        if sf is None:
            eta = self._predict_eta_np(X)
            r01 = (eta - eta.min()) / (eta.max() - eta.min() + 1e-12)
            return r01.astype(float)

        out = []
        t = float(t_years)
        for times, surv in sf:
            idx = np.searchsorted(times, t, side="right") - 1
            idx = int(np.clip(idx, 0, len(times) - 1))
            out.append(float(1.0 - surv[idx]))
        return np.asarray(out, dtype=float)
