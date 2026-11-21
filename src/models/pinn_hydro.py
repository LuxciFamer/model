"""Meta-PINN-Hydro hybrid model, neural-only ablation, and the shared sequence trainer."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import all_metrics
from src.models.losses import composite_loss
from src.physics.diff_loess_hydro import DiffLoessHydro


class MetaPINNHydro(nn.Module):
    """Hard physics core + GRU branch (bounded parameter modulation + bounded residual).

    Q_hat = Q_phys * exp(gamma * tanh(r)) keeps runoff non-negative and the residual
    bounded in log space (|log Q_hat - log Q_phys| <= gamma).
    """

    def __init__(self, n_feat, bounds, CN0, init_params, hidden=32, dropout=0.1, gamma0=0.3,
                 mod_et=0.3, mod_cn=0.25, use_inf=True, use_modulation=True, use_residual=True):
        super().__init__()
        self.phys = DiffLoessHydro(bounds, CN0, init=init_params, mod_et=mod_et, mod_cn=mod_cn,
                                   use_inf=use_inf)
        self.gru = nn.GRU(n_feat + 4, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.gamma = gamma0
        self.use_modulation, self.use_residual = use_modulation, use_residual

    def forward(self, X, P, T, E):
        with torch.no_grad():
            base = self.phys(P, T, E)
            sm = self.phys.params()["Smax"]
            st = torch.stack([torch.log1p(base["S0"][:, 0]), base["S1"][:, 0] / sm,
                              torch.log1p(base["Sf"][:, 0]), torch.log1p(base["Ss"][:, 0])], 1)
        h, _ = self.gru(torch.cat([X, st.float()], 1)[:, None, :])
        o = self.head(self.drop(h[:, 0, :])).double()
        m_et = o[:, 0:1] if self.use_modulation else None
        m_cn = o[:, 1:2] if self.use_modulation else None
        out = self.phys(P, T, E, m_et=m_et, m_cn=m_cn)
        res = self.gamma * torch.tanh(o[:, 2]) if self.use_residual else torch.zeros_like(o[:, 2])
        q_phys = out["Q"][:, 0]
        res_out = {k: v[:, 0] for k, v in out.items()}
        res_out["Qphys"] = q_phys
        res_out["Q"] = q_phys * torch.exp(res)
        res_out["res"] = res
        res_out["m_et"] = o[:, 0]; res_out["m_cn"] = o[:, 1]
        return res_out


class NeuralOnly(nn.Module):
    """Ablation 'w/o hard physics': GRU seq2seq with soft (monotonic recession) loss only."""

    def __init__(self, n_feat, hidden=32, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, P, T, E):
        h, _ = self.gru(X[:, None, :])
        q = nn.functional.softplus(self.head(self.drop(h[:, 0, :]))[:, 0]).double()
        return {"Q": q, "P": P}


def wbe_from_out(out, P, mask):
    """Closure error (% of P): physics conserves mass, so residual = sum(Qphys - Q_hat)."""
    if "Qphys" not in out:
        return float("nan")
    d = (out["Qphys"] - out["Q"]).detach().cpu().numpy()[mask].sum()
    return float(d / P[mask].sum() * 100)


def score(q_sim, obs: dict, mask, P, wbe=None):
    """Metrics on a period. obs: {'daily': (N,)} or {'month_idx','q_month'} (monthly means)."""
    if obs.get("daily") is not None:
        m = mask & np.isfinite(obs["daily"])
        return all_metrics(obs["daily"][m], q_sim[m], wbe=wbe) | {"MVR": _mvr(q_sim, P, mask)}
    mi, qm = obs["month_idx"], obs["q_month"]
    months = np.unique(mi[mask])
    full = [mm for mm in months if np.all(mask[mi == mm]) and np.isfinite(qm[mm])]
    sim = np.array([q_sim[mi == mm].mean() for mm in full])
    return all_metrics(qm[full], sim, wbe=wbe) | {"MVR": _mvr(q_sim, P, mask)}


def _mvr(q, P, mask):
    from src.evaluation.metrics import monotonic_violation_rate
    idx = np.flatnonzero(mask)
    return monotonic_violation_rate(q[idx], P[idx])


def j_score(m, coef):
    return 0.5 * m["KGE"] + 0.5 * m["NSE"] - coef * (abs(m["WBE"]) if np.isfinite(m["WBE"]) else 0.0)


def train_seq_model(model, X, P, T, E, obs, m_cal, m_val, lam, lr, epochs, patience,
                    j_coef=0.05, month=None, verbose=False):
    """Full-sequence training: loss on calibration days, early stopping on validation J.
    The GRU is causal and only data up to the end of validation are passed, so the
    test period is never seen during training/selection."""
    Pn = P.cpu().numpy()
    q_obs_t = torch.as_tensor(np.nan_to_num(obs["daily"]) if obs.get("daily") is not None
                              else np.zeros(len(Pn)), dtype=torch.float64)
    mask_t = torch.as_tensor(m_cal)
    if obs.get("daily") is not None:
        mask_t = mask_t & torch.as_tensor(np.isfinite(obs["daily"]))
        qscale = float(np.nanmean(obs["daily"][m_cal]))
    else:
        qscale = float(np.nanmean(obs["q_month"]))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best, best_state, wait, hist = -np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X, P, T, E)
        loss, terms = composite_loss(out, q_obs_t, mask_t, lam, qscale, month=month)
        if not torch.isfinite(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            out_v = model(X, P, T, E)
        q = out_v["Q"].cpu().numpy()
        mv = score(q, obs, m_val, Pn, wbe=wbe_from_out(out_v, Pn, m_val))
        js = j_score(mv, j_coef)
        loss_val = float(loss.detach())
        hist.append({"epoch": ep, "loss": loss_val, **terms, "val_KGE": mv["KGE"],
                     "val_NSE": mv["NSE"], "val_WBE": mv["WBE"], "val_J": js})
        if verbose and ep % 10 == 0:
            print(f"  ep{ep:3d} loss={loss_val:.4f} valNSE={mv['NSE']:.3f} valKGE={mv['KGE']:.3f}")
        if js > best + 1e-4:
            best, best_state, wait = js, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best
