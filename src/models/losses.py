"""Physics-informed composite loss for the hybrid model (daily or monthly targets)."""
from __future__ import annotations

import torch


def data_loss(q_hat, q_obs, mask, month_idx=None, q_month=None, month_mask=None):
    """0.65 * NSE-loss + 0.35 * log-MSE. Daily: q_obs (N,) with mask. Monthly: aggregate."""
    if month_idx is not None:
        nm = int(month_idx.max()) + 1
        sim_m = torch.zeros(nm, dtype=q_hat.dtype).index_add_(0, month_idx, q_hat)
        cnt = torch.zeros(nm, dtype=q_hat.dtype).index_add_(0, month_idx, torch.ones_like(q_hat))
        s, o = (sim_m / cnt)[month_mask], q_month[month_mask]
    else:
        s, o = q_hat[mask], q_obs[mask]
    eps = 0.01 * o.mean().detach() + 1e-6
    nse_l = ((s - o) ** 2).sum() / (((o - o.mean()) ** 2).sum() + 1e-9)
    log_l = ((torch.log(s + eps) - torch.log(o + eps)) ** 2).mean()
    return 0.65 * nse_l + 0.35 * log_l


def water_balance_loss(q_hat, q_phys, mask, qscale, window=30):
    """Residual volume between hybrid output and mass-conserving physics (30-d + global)."""
    d = (q_hat - q_phys)[mask]
    if d.numel() < window:
        return (d.sum() / (qscale * d.numel())) ** 2
    cs = torch.cat([d.new_zeros(1), torch.cumsum(d, 0)])
    w = cs[window:] - cs[:-window]
    return (w / (qscale * window)).pow(2).mean() + (d.sum() / (qscale * d.numel())).pow(2)


def ode_loss(q_hat, q_phys, mask, eps):
    """Storage-discharge (recession) consistency: d ln Q_hat/dt vs d ln Q_phys/dt."""
    lh = torch.log(q_hat + eps); lp = torch.log(q_phys + eps)
    r = (lh[1:] - lh[:-1]) - (lp[1:] - lp[:-1])
    return r[mask[1:]].pow(2).mean()


def mono_loss(q_hat, peff, mask, qscale, thr=0.2):
    dry = (peff[1:] < thr) & mask[1:]
    if dry.sum() == 0:
        return q_hat.new_zeros(())
    rise = torch.relu(q_hat[1:] - q_hat[:-1])[dry]
    return (rise / qscale).pow(2).mean()


def composite_loss(out, q_obs, mask, lam: dict, qscale, month=None):
    q_hat = out["Q"]
    terms = {}
    if month is not None:
        terms["data"] = data_loss(q_hat, None, None, *month)
    else:
        terms["data"] = data_loss(q_hat, q_obs, mask)
    eps = 0.01 * qscale
    if "Qphys" in out:
        terms["wb"] = water_balance_loss(q_hat, out["Qphys"], mask, qscale)
        terms["ode"] = ode_loss(q_hat, out["Qphys"], mask, eps)
        terms["res"] = out["res"][mask].pow(2).mean()
        peff = out["Peff"]
    else:
        z = q_hat.new_zeros(())
        terms["wb"] = z; terms["ode"] = z; terms["res"] = z
        peff = out["P"]
    terms["mono"] = mono_loss(q_hat, peff, mask, qscale)
    total = terms["data"] + lam.get("lambda_wb", 0) * terms["wb"] + lam.get("lambda_ode", 0) * terms["ode"] \
        + lam.get("lambda_mono", 0) * terms["mono"] + lam.get("lambda_res", 0) * terms["res"]
    return total, {k: float(v.detach()) for k, v in terms.items()}
