"""Differentiable Loess-Hydro core (PyTorch). Same equations as loess_hydro_np.

Physical parameters are learnable through a sigmoid map onto hydrologically
plausible bounds; optional time-varying modulations (from the neural branch)
act on the ET coefficient and the curve number.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.loess_hydro_np import PARAM_NAMES


@torch.jit.script
def _core(P: torch.Tensor, T: torch.Tensor, E: torch.Tensor,
          Tth: torch.Tensor, Tm: torch.Tensor, Df: torch.Tensor, cn_scale: torch.Tensor,
          lam_ia: torch.Tensor, Smax: torch.Tensor, Qmax: torch.Tensor, fpar: torch.Tensor,
          alpha_et: torch.Tensor, kf: torch.Tensor, ks: torch.Tensor,
          m_et: torch.Tensor, m_cn: torch.Tensor, CN0: float, tau_T: float,
          mod_et: float, mod_cn: float, use_inf: bool,
          S0: torch.Tensor, S1: torch.Tensor, Sf: torch.Tensor, Ss: torch.Tensor):
    # P,T,E,m_et,m_cn: (N,B); params: (B,)
    N = P.shape[0]
    outs = []
    for t in range(N):
        p_t = P[t]
        T_t = T[t]
        a_et = alpha_et * (1.0 + mod_et * torch.tanh(m_et[t]))
        cn_m = 1.0 + mod_cn * torch.tanh(m_cn[t])
        psnow = p_t * torch.sigmoid((Tth - T_t) / tau_T)
        s0a = S0 + psnow
        melt_pot = Df * F.softplus(T_t - Tm)
        M = torch.minimum(s0a, melt_pot)
        S0 = s0a - M
        peff = p_t - psnow + M
        if use_inf:
            CN = torch.clamp(CN0 * cn_scale * cn_m, 30.0, 98.0)
            spot = 25400.0 / CN - 254.0
            ia = lam_ia * spot * (1.0 - 0.5 * torch.clamp(S1 / Smax, 0.0, 1.0))
            x = torch.relu(peff - ia)
            qinf = torch.minimum(x * x / (x + spot + 1e-9), peff)
        else:
            qinf = torch.zeros_like(peff)
        fin = peff - qinf
        s1a = S1 + fin
        ea = torch.minimum(a_et * E[t] * torch.clamp(s1a / Smax, 0.0, 1.0), s1a)
        qb = torch.minimum(Qmax * torch.exp(-fpar * torch.relu(Smax - s1a)), s1a - ea)
        qsat = torch.relu(s1a - ea - qb - Smax)
        S1 = s1a - ea - qb - qsat
        sfa = Sf + qinf + qsat
        qf = kf * sfa
        Sf = sfa - qf
        ssa = Ss + qb
        qs = ks * ssa
        Ss = ssa - qs
        outs.append(torch.stack([qf + qs, ea, S0, S1, Sf, Ss, qinf, qsat, qb, peff], 0))
    return torch.stack(outs, 0)  # (N, 10, B)


OUT_KEYS = ["Q", "Ea", "S0", "S1", "Sf", "Ss", "Qinf", "Qsat", "Qb", "Peff"]


class DiffLoessHydro(nn.Module):
    def __init__(self, bounds: dict, CN0: float, init: dict | None = None, tau_T: float = 1.0,
                 mod_et: float = 0.3, mod_cn: float = 0.25, use_inf: bool = True,
                 learnable: bool = True):
        super().__init__()
        lo = torch.tensor([bounds[k][0] for k in PARAM_NAMES], dtype=torch.float64)
        hi = torch.tensor([bounds[k][1] for k in PARAM_NAMES], dtype=torch.float64)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        if init is None:
            u = torch.full((len(PARAM_NAMES),), 0.5, dtype=torch.float64)
        else:
            v = torch.tensor([init[k] for k in PARAM_NAMES], dtype=torch.float64)
            u = ((v - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)
        logit = torch.log(u / (1 - u))
        self.logit = nn.Parameter(logit, requires_grad=learnable)
        self.CN0, self.tau_T, self.mod_et, self.mod_cn, self.use_inf = CN0, tau_T, mod_et, mod_cn, use_inf

    def params(self) -> dict:
        v = self.lo + (self.hi - self.lo) * torch.sigmoid(self.logit)
        return {k: v[i] for i, k in enumerate(PARAM_NAMES)}

    def params_numpy(self) -> dict:
        return {k: float(v.detach().cpu()) for k, v in self.params().items()}

    def forward(self, P, T, E, m_et=None, m_cn=None, init_state=None):
        """P,T,E: (N,) or (N,B) float64 tensors. Returns dict of (N,B) tensors."""
        if P.dim() == 1:
            P, T, E = P[:, None], T[:, None], E[:, None]
        N, B = P.shape
        prm = self.params()
        pv = [prm[k].expand(B) for k in PARAM_NAMES]
        if m_et is None:
            m_et = torch.zeros_like(P)
        if m_cn is None:
            m_cn = torch.zeros_like(P)
        if init_state is None:
            z = torch.zeros(B, dtype=P.dtype, device=P.device)
            S0, S1, Sf, Ss = z, 0.5 * pv[5], z, z
        else:
            S0, S1, Sf, Ss = init_state
        out = _core(P, T, E, *pv, m_et, m_cn, float(self.CN0), float(self.tau_T),
                    float(self.mod_et), float(self.mod_cn), bool(self.use_inf), S0, S1, Sf, Ss)
        return {k: out[:, i, :] for i, k in enumerate(OUT_KEYS)}


def to_t(x, dtype=torch.float64, device="cpu"):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
