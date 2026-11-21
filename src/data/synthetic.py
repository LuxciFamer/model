"""Synthetic identical-twin runoff (ONLY for pipeline smoke tests without observations).

Truth = Loess-Hydro core with known parameters + two time-varying processes that a
fixed-parameter conceptual model cannot represent:
  (1) seasonal ET coefficient (vegetation phenology): alpha(t) = a0 (1 + 0.25 sin(2pi(DOY-120)/365))
  (2) storm-intensity dependent curve number: CN(t) = CN0 (1 + 0.15 tanh((P_max_sta - 20)/10))
plus multiplicative log-normal noise (sigma = 0.10). alpha_et is bisected so that the
2016-2023 runoff coefficient matches ~0.075 (Daning long-term natural runoff 1.84e8 m3/yr
over ~3992 km2 ≈ 46 mm/yr vs ≈ 620 mm/yr precipitation).
These numbers are NOT observations; results obtained with them verify the pipeline only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics.loess_hydro_np import run_np

TRUE_PARAMS = {"Tth": 0.5, "Tm": 1.0, "Df": 2.5, "cn_scale": 1.0, "lam_ia": 0.15,
               "Smax": 180.0, "Qmax": 0.4, "f": 0.03, "alpha_et": 0.75, "kf": 0.45, "ks": 0.02}
TARGET_RC = 0.075
NOISE_SIGMA = 0.10


def truth_modulations(df: pd.DataFrame, mod_et=0.3, mod_cn=0.25):
    doy = df.index.dayofyear.values
    fac_et = 0.25 * np.sin(2 * np.pi * (doy - 120) / 365.0)
    fac_cn = 0.15 * np.tanh((df["P_max_sta"].values - 20.0) / 10.0)
    m_et = np.arctanh(np.clip(fac_et / mod_et, -0.999, 0.999))
    m_cn = np.arctanh(np.clip(fac_cn / mod_cn, -0.999, 0.999))
    return m_et[:, None], m_cn[:, None]


def make_twin(df: pd.DataFrame, CN0: float, area_km2: float, seed: int = 2024):
    P, T, E = df["P"].values, df["Tmean"].values, df["ET0"].values
    m_et, m_cn = truth_modulations(df)
    sel = df.index >= "2016-01-01"
    prm = dict(TRUE_PARAMS)

    def rc(a):
        prm["alpha_et"] = a
        q = run_np(P, T, E, {k: np.array([v]) for k, v in prm.items()}, CN0, m_et=m_et, m_cn=m_cn)[:, 0]
        return q[sel].sum() / P[sel].sum(), q

    lo, hi = 0.40, 1.30
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r, _ = rc(mid)
        if r > TARGET_RC:
            lo = mid
        else:
            hi = mid
    r, q = rc(0.5 * (lo + hi))
    rng = np.random.default_rng(seed)
    q_obs = q * np.exp(rng.normal(0, NOISE_SIGMA, len(q)) - 0.5 * NOISE_SIGMA ** 2)
    out = pd.DataFrame({"date": df.index, "q_obs_m3s": q_obs * area_km2 / 86.4})
    info = {"true_params": dict(prm), "runoff_coefficient_2016_2023": float(r),
            "noise_sigma": NOISE_SIGMA, "seed": seed, "target_rc": TARGET_RC,
            "truth_processes": ["seasonal alpha_et (+-25%)", "storm-intensity CN (+-15%) via P_max_sta"],
            "mean_runoff_mm_per_yr": float(q_obs[sel].sum() / (sel.sum() / 365.25))}
    return out, info
