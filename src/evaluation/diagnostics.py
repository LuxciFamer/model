"""Physical state diagnostics, gradient-based factor attribution, and metrics summary."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.optimization.bilevel_tuner import Context, build_model
from src.utils import p

METRIC_COLS = ["NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta",
               "R2", "RMSE", "MAE", "PBIAS", "FHV", "FLV", "WBE", "MVR"]


def compute_state_diagnostics(df_feat: pd.DataFrame, states_df: pd.DataFrame,
                              m_test: np.ndarray, m_ext: np.ndarray) -> dict:
    """Evaluate physical sanity, component partitioning, and internal state consistency."""
    idx = df_feat.index
    m_nonwarm = np.asarray(idx >= "2016-01-01")
    periods = {"non_warmup_2016_2023": m_nonwarm, "test_2021_2023": m_test, "extreme_2021": m_ext}
    out = {}
    P = df_feat["P"].values
    ET0 = df_feat["ET0"].values
    API15 = df_feat["API_15"].values
    S1 = states_df["S1"].values
    Ea = states_df["Ea"].values
    Q = states_df["Q"].values
    Qphys = states_df["Qphys"].values
    Qinf = states_df["Qinf"].values
    Qsat = states_df["Qsat"].values
    Qb = states_df["Qb"].values
    S_tot = states_df["S0"].values + states_df["S1"].values + states_df["Sf"].values + states_df["Ss"].values

    for name, m in periods.items():
        n_years = max(1e-6, m.sum() / 365.25)
        r_s1, p_s1 = spearmanr(S1[m], API15[m])
        r_ea, p_ea = spearmanr(Ea[m], ET0[m])
        q_gen = float(Qinf[m].sum() + Qsat[m].sum() + Qb[m].sum())
        denom = q_gen if q_gen > 1e-12 else 1.0
        q_sum = float(Q[m].sum())
        res_rel = float(np.abs(Q[m] - Qphys[m]).sum() / max(q_sum, 1e-12) * 100.0)
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        s_start = S_tot[i0 - 1] if i0 > 0 else S_tot[i0]
        wb_phys_resid_mm = float(P[m].sum() - Ea[m].sum() - Qphys[m].sum() - (S_tot[i1] - s_start))
        wb_hybrid_resid_mm = float(P[m].sum() - Ea[m].sum() - Q[m].sum() - (S_tot[i1] - s_start))
        out[name] = {
            "n_days": int(m.sum()),
            "P_mm_yr": float(P[m].sum() / n_years),
            "ET0_mm_yr": float(ET0[m].sum() / n_years),
            "Ea_mm_yr": float(Ea[m].sum() / n_years),
            "Q_mm_yr": float(Q[m].sum() / n_years),
            "Qphys_mm_yr": float(Qphys[m].sum() / n_years),
            "runoff_coeff": float(Q[m].sum() / max(P[m].sum(), 1e-12)),
            "Qinf_pct": float(Qinf[m].sum() / denom * 100.0),
            "Qsat_pct": float(Qsat[m].sum() / denom * 100.0),
            "Qb_pct": float(Qb[m].sum() / denom * 100.0),
            "neural_residual_rel_pct": res_rel,
            "physics_dominance_pct": float(max(0.0, 100.0 - res_rel)),
            "spearman_S1_API15": float(r_s1),
            "pval_S1_API15": float(p_s1),
            "spearman_Ea_ET0": float(r_ea),
            "pval_Ea_ET0": float(p_ea),
            "wb_phys_resid_mm": wb_phys_resid_mm,
            "wb_hybrid_resid_mm": wb_hybrid_resid_mm,
            "wb_hybrid_resid_pct": float(wb_hybrid_resid_mm / max(P[m].sum(), 1e-12) * 100.0),
            "min_Q_mm": float(Q[m].min()),
            "min_Q_nonneg": bool(Q[m].min() >= 0.0),
            "ea_within_1p3_et0_pct": float(np.mean(Ea[m] <= 1.3 * ET0[m] + 1e-6) * 100.0),
            "s1_api15_positive": bool(r_s1 > 0.0),
        }
    return out


def compute_gradient_attribution(ctx: Context, hp: dict, priors: dict, seeds: list[int]) -> pd.DataFrame:
    """Compute |dQ_hat / dX_t| sensitivity averaged over the test period and ensemble seeds."""
    X_base, P, T, E = ctx.tensors()
    mask = torch.as_tensor(ctx.m_test)
    per_seed = []
    for s in seeds:
        ckpt = p(f"results/checkpoints/MetaPINN_seed{s}.pt")
        if not ckpt.exists():
            continue
        model = build_model(ctx, hp, priors, "full")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        X_in = X_base.clone().detach().requires_grad_(True)
        out = model(X_in, P, T, E)
        out["Q"][mask].sum().backward()
        sens = X_in.grad[mask].abs().mean(dim=0).detach().numpy()
        sens_pct = sens / max(float(sens.sum()), 1e-12) * 100.0
        per_seed.append(sens_pct)
    arr = np.stack(per_seed, axis=0)
    df = pd.DataFrame({
        "factor": ctx.features,
        "grad_importance_pct": arr.mean(axis=0),
        "grad_importance_std": arr.std(axis=0),
    }).sort_values("grad_importance_pct", ascending=False).reset_index(drop=True)
    return df


def build_summary_table(baselines_data: dict, final_data: dict) -> pd.DataFrame:
    """Flatten all baseline and Meta-PINN-Hydro metrics into a clean summary DataFrame."""
    rows = []
    models = dict(baselines_data.get("metrics", {}))
    models["Meta-PINN-Hydro"] = final_data["ensemble"]
    for model_name, periods in models.items():
        for per_name, m in periods.items():
            row = {"model": model_name, "period": per_name}
            for col in METRIC_COLS:
                row[col] = m.get(col, np.nan)
            row["n"] = m.get("n", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)
