import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.physics.loess_hydro_np import PARAM_NAMES, run_np  # noqa: E402
from src.physics.diff_loess_hydro import DiffLoessHydro  # noqa: E402

BOUNDS = {"Tth": [-2, 2], "Tm": [-1, 3], "Df": [0.5, 6], "cn_scale": [0.75, 1.25],
          "lam_ia": [0.05, 0.3], "Smax": [50, 500], "Qmax": [0.01, 3], "f": [0.005, 0.1],
          "alpha_et": [0.4, 1.3], "kf": [0.1, 0.9], "ks": [0.001, 0.1]}


def _forcing(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    P = np.where(rng.random(n) < 0.25, rng.gamma(0.8, 8.0, n), 0.0)
    T = 10 + 14 * np.sin(2 * np.pi * (t - 100) / 365) + rng.normal(0, 3, n)
    E = np.clip(2.5 + 2.0 * np.sin(2 * np.pi * (t - 100) / 365), 0.2, None)
    return P, T, E


def _rand_params(rng):
    return {k: float(rng.uniform(*BOUNDS[k])) for k in PARAM_NAMES}


def test_numpy_torch_equivalence():
    P, T, E = _forcing()
    rng = np.random.default_rng(1)
    for _ in range(5):
        prm = _rand_params(rng)
        q_np = run_np(P, T, E, {k: np.array([v]) for k, v in prm.items()}, 75.0)[:, 0]
        m = DiffLoessHydro(BOUNDS, 75.0, init=prm)
        pt = m.params_numpy()
        q_np2 = run_np(P, T, E, {k: np.array([v]) for k, v in pt.items()}, 75.0)[:, 0]
        with torch.no_grad():
            q_t = m(torch.tensor(P), torch.tensor(T), torch.tensor(E))["Q"][:, 0].numpy()
        assert np.max(np.abs(q_t - q_np2)) < 1e-5
        assert np.max(np.abs(q_np - q_np2)) < 1e-2  # logit round-trip of init params


def test_mass_conservation_numpy():
    P, T, E = _forcing()
    rng = np.random.default_rng(2)
    for _ in range(5):
        prm = {k: np.array([v]) for k, v in _rand_params(rng).items()}
        Q, st = run_np(P, T, E, prm, 75.0, return_states=True)
        S_end = st["S0"][-1] + st["S1"][-1] + st["Sf"][-1] + st["Ss"][-1]
        S_start = 0.5 * prm["Smax"]
        resid = P.sum() - st["Ea"].sum() - Q.sum() - (S_end - S_start)
        assert abs(float(resid[0])) < 1e-3
        assert (Q >= 0).all() and (st["Ea"] >= -1e-12).all()


def test_gradients_flow():
    P, T, E = _forcing(400)
    m = DiffLoessHydro(BOUNDS, 75.0)
    Pt, Tt, Et = map(torch.tensor, (P, T, E))
    m_et = torch.zeros(len(P), 1, dtype=torch.float64, requires_grad=True)
    m_cn = torch.zeros(len(P), 1, dtype=torch.float64, requires_grad=True)
    out = m(Pt, Tt, Et, m_et=m_et, m_cn=m_cn)
    loss = (out["Q"] ** 2).mean() + out["Ea"].mean()
    loss.backward()
    g = m.logit.grad
    assert g is not None and torch.isfinite(g).all() and (g.abs() > 0).sum() >= 6
    assert m_et.grad is not None and torch.isfinite(m_et.grad).all() and m_et.grad.abs().sum() > 0
    assert m_cn.grad is not None and m_cn.grad.abs().sum() > 0


def test_no_infiltration_branch_flag():
    P, T, E = _forcing(300)
    prm = {k: np.array([v]) for k, v in _rand_params(np.random.default_rng(3)).items()}
    _, st = run_np(P, T, E, prm, 75.0, use_inf=False, return_states=True)
    assert np.all(st["Qinf"] == 0)
