"""Shared experiment context + outer-loop-II (bi-level) tuner for Meta-PINN-Hydro."""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import FeatureScaler, load_features, load_runoff, monthly_index, period_mask
from src.data.meteo_et0 import FEATURES_ALL, STORM_FEATURES
from src.models.pinn_hydro import MetaPINNHydro, NeuralOnly, score, train_seq_model, wbe_from_out
from src.utils import load_cfg, load_json, load_meta, seed_all

HIDDEN_CHOICES = [16, 24, 32, 48, 64]
SPACE_KEYS = ["log10_lambda_wb", "log10_lambda_ode", "log10_lambda_mono", "log10_lambda_res",
              "hidden_dim", "log10_lr", "dropout", "gamma0"]
J_COEF = 0.05  # J = 0.5 KGE + 0.5 NSE - J_COEF * |WBE(% of P)|


class Context:
    """Loads features/targets once; builds tensors for train (<= end of validation) and full."""

    def __init__(self, calib_period=None, features=None):
        self.meta, self.cfg = load_meta(), load_cfg()
        self.static = load_json("data/processed/xinshui_static_attributes.json")
        self.df = load_features()
        self.run = load_runoff(self.meta, self.cfg)
        sp = self.cfg["split"]
        idx = self.df.index
        self.idx = idx
        self.m_cal = period_mask(idx, calib_period or sp["calib"])
        self.m_val = period_mask(idx, sp["valid"])
        self.m_test = period_mask(idx, sp["test"])
        self.m_ext = period_mask(idx, sp["extreme_year"])
        self.n_train = int(np.flatnonzero(self.m_val).max()) + 1  # rows up to end of validation
        self.features = features or FEATURES_ALL
        Xraw = self.df[self.features].values
        self.scaler = FeatureScaler().fit(Xraw, self.m_cal)
        self.Xs = self.scaler.transform(Xraw).astype(np.float32)
        self.P = self.df["P"].values; self.T = self.df["Tmean"].values; self.E = self.df["ET0"].values
        if self.run["daily"] is not None:
            self.q = self.run["daily"].reindex(idx).values.astype(float)
            self.obs = {"daily": self.q}
            self.month = None
        else:
            mi = monthly_index(idx)
            qm = np.full(mi.max() + 1, np.nan)
            ser = self.run["monthly"]
            for d, v in ser.items():
                k = (d.year * 12 + d.month) - (idx[0].year * 12 + idx[0].month)
                if 0 <= k < len(qm):
                    qm[k] = v
            self.q = None
            self.obs = {"month_idx": mi, "q_month": qm}
            self.month = mi
        self.CN0 = float(self.static["CN0"])
        self.bounds = self.cfg["physics"]["bounds"]

    def tensors(self, n=None):
        n = n or len(self.P)
        f64 = lambda a: torch.as_tensor(a[:n], dtype=torch.float64)  # noqa: E731
        return torch.as_tensor(self.Xs[:n]), f64(self.P), f64(self.T), f64(self.E)

    def obs_slice(self, n):
        if "daily" in self.obs:
            return {"daily": self.obs["daily"][:n]}
        return {"month_idx": self.obs["month_idx"][:n], "q_month": self.obs["q_month"]}

    def month_args(self, n, mask):
        if self.month is None:
            return None
        mi = torch.as_tensor(self.month[:n])
        qm = torch.as_tensor(np.nan_to_num(self.obs["q_month"]), dtype=torch.float64)
        months = np.unique(self.month[:n][mask[:n]])
        mm = np.zeros(len(qm), bool)
        for k in months:
            if np.all(mask[:n][self.month[:n] == k]) and np.isfinite(self.obs["q_month"][k]):
                mm[k] = True
        return (mi, qm, torch.as_tensor(mm))


def decode(x) -> dict:
    x = np.asarray(x, float)
    h = HIDDEN_CHOICES[int(np.argmin([abs(c - x[4]) for c in HIDDEN_CHOICES]))]
    return {"lambda_wb": 10 ** x[0], "lambda_ode": 10 ** x[1], "lambda_mono": 10 ** x[2],
            "lambda_res": 10 ** x[3], "hidden_dim": h, "lr": 10 ** x[5], "dropout": float(x[6]),
            "gamma0": float(x[7])}


def space_bounds(cfg):
    ss = cfg["stage2"]["search_space"]
    return np.array([ss[k][0] for k in SPACE_KEYS]), np.array([ss[k][1] for k in SPACE_KEYS])


def build_model(ctx: Context, hp: dict, priors: dict, variant: str = "full"):
    hc = ctx.cfg["hybrid"]
    if variant == "no_physics":
        return NeuralOnly(len(ctx.features), hp["hidden_dim"], hp["dropout"])
    return MetaPINNHydro(len(ctx.features), ctx.bounds, ctx.CN0, priors, hidden=hp["hidden_dim"],
                         dropout=hp["dropout"], gamma0=hp["gamma0"], mod_et=hc["modulation_et"],
                         mod_cn=hc["modulation_cn"], use_inf=(variant != "no_inf"),
                         use_modulation=(variant != "no_modulation"))


def fit(ctx: Context, hp: dict, priors: dict, epochs: int, patience: int, seed: int,
        variant: str = "full", verbose=False):
    seed_all(seed)
    model = build_model(ctx, hp, priors, variant)
    lam = {k: hp[k] for k in ["lambda_wb", "lambda_ode", "lambda_mono", "lambda_res"]}
    if variant == "no_soft":
        lam.update(lambda_wb=0.0, lambda_ode=0.0, lambda_mono=0.0)
    n = ctx.n_train
    X, P, T, E = ctx.tensors(n)
    month = ctx.month_args(n, ctx.m_cal)
    model, hist, best = train_seq_model(model, X, P, T, E, ctx.obs_slice(n), ctx.m_cal[:n],
                                        ctx.m_val[:n], lam, hp["lr"], epochs, patience,
                                        j_coef=J_COEF, month=month, verbose=verbose)
    return model, hist, best


def predict_full(ctx: Context, model):
    model.eval()
    X, P, T, E = ctx.tensors()
    with torch.no_grad():
        out = model(X, P, T, E)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def evaluate_periods(ctx: Context, q_sim, out=None):
    res = {}
    for name, m in [("calib", ctx.m_cal), ("valid", ctx.m_val), ("test", ctx.m_test), ("extreme_2021", ctx.m_ext)]:
        wbe = None
        if out is not None and "Qphys" in out:
            wbe = float((out["Qphys"][m] - out["Q"][m]).sum() / ctx.P[m].sum() * 100)
        res[name] = score(q_sim, ctx.obs, m, ctx.P, wbe=wbe)
    return res


def val_eval(ctx, hp, priors, epochs, patience, seed):
    """Proxy evaluation for outer loop II (validation-period metrics only)."""
    model, hist, best = fit(ctx, hp, priors, epochs, patience, seed)
    X, P, T, E = ctx.tensors(ctx.n_train)
    model.eval()
    with torch.no_grad():
        out = model(X, P, T, E)
    q = out["Q"].numpy()
    m = score(q, ctx.obs_slice(ctx.n_train), ctx.m_val[:ctx.n_train], ctx.P[:ctx.n_train],
              wbe=wbe_from_out(out, ctx.P[:ctx.n_train], ctx.m_val[:ctx.n_train]))
    return m
