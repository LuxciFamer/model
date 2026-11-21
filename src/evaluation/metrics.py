"""Hydrological performance metrics."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def _clean(obs, sim):
    obs = np.asarray(obs, float); sim = np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim):
    o, s = _clean(obs, sim)
    return float(1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def kge(obs, sim):
    o, s = _clean(obs, sim)
    r = np.corrcoef(o, s)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), float(r), float(alpha), float(beta)


def fhv(obs, sim, h=0.02):
    """Peak-flow bias of the top h exceedance segment (Yilmaz et al., 2008), %."""
    o, s = _clean(obs, sim)
    so, ss = np.sort(o)[::-1], np.sort(s)[::-1]
    n = max(1, int(np.ceil(h * len(o))))
    return float((ss[:n].sum() - so[:n].sum()) / so[:n].sum() * 100)


def flv(obs, sim, l=0.3):
    """Low-flow bias of the bottom l segment in log space (Yilmaz et al., 2008), %."""
    o, s = _clean(obs, sim)
    so, ss = np.log(np.sort(o) + EPS), np.log(np.sort(s) + EPS)
    n = max(2, int(np.ceil(l * len(o))))
    qo, qs = so[:n], ss[:n]
    so_sum = np.sum(qo - qo.min())
    ss_sum = np.sum(qs - qs.min())
    if so_sum <= 0:
        return float("nan")
    return float(-1 * (ss_sum - so_sum) / so_sum * 100)


def monotonic_violation_rate(sim, P, thr=0.2, tol=1e-3):
    """Fraction (%) of dry days (P<thr) on which simulated flow rises."""
    sim = np.asarray(sim, float); P = np.asarray(P, float)
    dry = (P[1:] < thr)
    rise = sim[1:] > sim[:-1] * (1 + tol) + 1e-9
    if dry.sum() == 0:
        return float("nan")
    return float((rise & dry).sum() / dry.sum() * 100)


def all_metrics(obs, sim, P=None, wbe=None) -> dict:
    o, s = _clean(obs, sim)
    k, r, a, b = kge(o, s)
    out = {
        "NSE": nse(o, s), "KGE": k, "KGE_r": r, "KGE_alpha": a, "KGE_beta": b,
        "R2": float(np.corrcoef(o, s)[0, 1] ** 2),
        "RMSE": float(np.sqrt(np.mean((s - o) ** 2))),
        "MAE": float(np.mean(np.abs(s - o))),
        "PBIAS": float((s.sum() - o.sum()) / o.sum() * 100),
        "FHV": fhv(o, s), "FLV": flv(o, s),
        "n": int(len(o)),
    }
    if P is not None:
        out["MVR"] = monotonic_violation_rate(sim, P)
    out["WBE"] = float(wbe) if wbe is not None else float("nan")
    return out


def water_balance_error(P, Ea, Q, S_start, S_end):
    """Relative closure error (%) of P = Ea + Q + dS over a period."""
    P = np.sum(P); resid = P - np.sum(Ea) - np.sum(Q) - (S_end - S_start)
    return float(resid / P * 100)
