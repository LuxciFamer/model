"""Loess-Hydro conceptual core (NumPy, vectorised over a batch/population).

State: S0 snow, S1 soil, Sf fast reservoir, Ss slow reservoir (all mm).
Per step (daily): snow split/melt -> SCS-CN infiltration-excess (Q_inf)
-> soil: ET (Ea), baseflow generation (Qb), saturation excess (Q_sat)
-> two linear reservoirs routing -> Q_phys.
Mass is conserved exactly: sum(P) = sum(Ea) + sum(Q) + dS.
"""
from __future__ import annotations

import numpy as np

PARAM_NAMES = ["Tth", "Tm", "Df", "cn_scale", "lam_ia", "Smax", "Qmax", "f", "alpha_et", "kf", "ks"]


def bounds_array(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([bounds[k][0] for k in PARAM_NAMES], dtype=float)
    hi = np.array([bounds[k][1] for k in PARAM_NAMES], dtype=float)
    return lo, hi


def vec_to_params(x: np.ndarray) -> dict:
    x = np.atleast_2d(x)
    return {k: x[:, i] for i, k in enumerate(PARAM_NAMES)}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_np(P, T, ET0, params: dict, CN0: float, tau_T: float = 1.0,
           m_et=None, m_cn=None, mod_et: float = 0.3, mod_cn: float = 0.25,
           use_inf: bool = True, init_state=None, return_states: bool = False):
    """P, T, ET0: (N,) arrays. params: dict of (B,) arrays. Returns Q (N,B) [+ states]."""
    N = len(P)
    B = len(params["Smax"])
    g = {k: np.asarray(v, dtype=float).reshape(B) for k, v in params.items()}
    if init_state is None:
        S0 = np.zeros(B); S1 = 0.5 * g["Smax"]; Sf = np.zeros(B); Ss = np.zeros(B)
    else:
        S0, S1, Sf, Ss = [np.array(s, dtype=float).reshape(B) for s in init_state]
    Q = np.zeros((N, B))
    if return_states:
        keys = ["S0", "S1", "Sf", "Ss", "Ea", "Qinf", "Qsat", "Qb", "Qf", "Qs", "Peff", "M"]
        rec = {k: np.zeros((N, B)) for k in keys}
    for t in range(N):
        p_t, T_t, e_t = P[t], T[t], ET0[t]
        a_et = g["alpha_et"] * (1 + mod_et * np.tanh(m_et[t])) if m_et is not None else g["alpha_et"]
        cn_m = (1 + mod_cn * np.tanh(m_cn[t])) if m_cn is not None else 1.0
        # snow
        psnow = p_t * _sigmoid((g["Tth"] - T_t) / tau_T)
        s0a = S0 + psnow
        xm = T_t - g["Tm"]
        melt_pot = g["Df"] * np.where(xm > 20.0, xm, np.log1p(np.exp(np.minimum(xm, 20.0))))
        M = np.minimum(s0a, melt_pot)
        S0 = s0a - M
        peff = p_t - psnow + M
        # infiltration excess (SCS-CN, antecedent-moisture adjusted initial abstraction)
        if use_inf:
            CN = np.clip(CN0 * g["cn_scale"] * cn_m, 30.0, 98.0)
            spot = 25400.0 / CN - 254.0
            ia = g["lam_ia"] * spot * (1 - 0.5 * np.clip(S1 / g["Smax"], 0, 1))
            x = np.maximum(peff - ia, 0.0)
            qinf = np.minimum(x * x / (x + spot + 1e-9), peff)
        else:
            qinf = np.zeros(B)
        fin = peff - qinf
        # soil
        s1a = S1 + fin
        ea = np.minimum(a_et * e_t * np.clip(s1a / g["Smax"], 0, 1), s1a)
        qb = np.minimum(g["Qmax"] * np.exp(-g["f"] * np.maximum(g["Smax"] - s1a, 0.0)), s1a - ea)
        qsat = np.maximum(s1a - ea - qb - g["Smax"], 0.0)
        S1 = s1a - ea - qb - qsat
        # routing
        sfa = Sf + qinf + qsat
        qf = g["kf"] * sfa
        Sf = sfa - qf
        ssa = Ss + qb
        qs = g["ks"] * ssa
        Ss = ssa - qs
        Q[t] = qf + qs
        if return_states:
            for k, v in zip(keys, [S0, S1, Sf, Ss, ea, qinf, qsat, qb, qf, qs, peff, M]):
                rec[k][t] = v
    if return_states:
        return Q, rec
    return Q
