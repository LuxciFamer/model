"""Phase 3-4: Stage-I physics calibration (PSO/GWO/SSA) + baselines (SWAT, conceptual, LSTM, GRU, Base-PINN)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.dataset import load_swat  # noqa: E402
from src.evaluation.metrics import water_balance_error  # noqa: E402
from src.models.baselines import predict_seq2one, train_seq2one  # noqa: E402
from src.models.pinn_hydro import score  # noqa: E402
from src.optimization.bilevel_tuner import Context  # noqa: E402
from src.optimization.jobs import run_parallel  # noqa: E402
from src.optimization.metaheuristics import minimize  # noqa: E402
from src.physics.loess_hydro_np import PARAM_NAMES, bounds_array, run_np, vec_to_params  # noqa: E402
from src.utils import budget, p, save_json, seed_all  # noqa: E402


def batch_scores(Q, ctx: Context, mask):
    """Vectorised NSE/KGE for a population of simulations Q (n, B) on `mask`."""
    if ctx.month is None:
        sel = mask[:Q.shape[0]] & np.isfinite(ctx.q[:Q.shape[0]])
        o, s = ctx.q[:Q.shape[0]][sel], Q[sel]
    else:
        mi = ctx.month[:Q.shape[0]]; qm = ctx.obs["q_month"]
        months = [k for k in np.unique(mi[mask[:Q.shape[0]]])
                  if np.all(mask[:Q.shape[0]][mi == k]) and np.isfinite(qm[k])]
        s = np.stack([Q[mi == k].mean(0) for k in months]); o = qm[months]
    o = o[:, None]
    nse = 1 - ((s - o) ** 2).sum(0) / ((o - o.mean()) ** 2).sum()
    so, ss = o.std(), s.std(0)
    cov = ((s - s.mean(0)) * (o - o.mean())).mean(0)
    r = cov / (ss * so + 1e-12)
    kge = 1 - np.sqrt((r - 1) ** 2 + (ss / so - 1) ** 2 + (s.mean(0) / o.mean() - 1) ** 2)
    return nse, kge


def physics_periods(ctx, prm):
    Q, st = run_np(ctx.P, ctx.T, ctx.E, {k: np.array([v]) for k, v in prm.items()}, ctx.CN0,
                   return_states=True)
    q = Q[:, 0]
    S = st["S0"][:, 0] + st["S1"][:, 0] + st["Sf"][:, 0] + st["Ss"][:, 0]
    S_init = 0.5 * prm["Smax"]
    res = {}
    for name, m in [("calib", ctx.m_cal), ("valid", ctx.m_val), ("test", ctx.m_test), ("extreme_2021", ctx.m_ext)]:
        i0, i1 = np.flatnonzero(m)[[0, -1]]
        s0 = S[i0 - 1] if i0 > 0 else S_init
        wbe = water_balance_error(ctx.P[m], st["Ea"][m, 0], q[m], s0, S[i1])
        res[name] = score(q, ctx.obs, m, ctx.P, wbe=wbe)
    return q, st, res


def stage1(ctx: Context, cfg) -> dict:
    b = budget(cfg, "stage1")
    lo, hi = bounds_array(cfg["physics"]["bounds"])
    n = ctx.n_train

    def obj(X):
        Q = run_np(ctx.P[:n], ctx.T[:n], ctx.E[:n], vec_to_params(X), ctx.CN0)
        nse, kge = batch_scores(Q, ctx, ctx.m_cal)
        return -(0.5 * nse + 0.5 * kge)

    runs, best_overall = [], None
    for name in cfg["stage1"]["optimizers"]:
        for rep in range(b["repeats"]):
            t0 = time.time()
            r = minimize(name, obj, lo, hi, pop=b["pop"], iters=b["iters"], seed=100 * rep + 7)
            prm = dict(zip(PARAM_NAMES, map(float, r["x"])))
            Qv = run_np(ctx.P[:n], ctx.T[:n], ctx.E[:n], {k: np.array([v]) for k, v in prm.items()}, ctx.CN0)
            nse_v, kge_v = batch_scores(Qv, ctx, ctx.m_val)
            rec = {"optimizer": name, "repeat": rep, "best_cost": r["f"], "params": prm,
                   "history": r["history"], "n_evals": r["n_evals"], "seconds": time.time() - t0,
                   "valid_NSE": float(nse_v[0]), "valid_KGE": float(kge_v[0])}
            runs.append(rec)
            print(f"  {name} rep{rep}: calib J={-r['f']:.4f}  valid NSE={nse_v[0]:.3f}  ({rec['seconds']:.1f}s)")
            # select priors with the VALIDATION period (no test data)
            score_v = 0.5 * nse_v[0] + 0.5 * kge_v[0]
            if best_overall is None or score_v > best_overall["score_valid"]:
                best_overall = {"optimizer": name, "repeat": rep, "best_params": prm, "score_valid": float(score_v)}
    save_json({"runs": runs}, "results/metrics/stage1_physics_calibration.json")
    save_json(best_overall, "results/checkpoints/stage1_physics_priors.json")
    return {"runs": runs, "best": best_overall}


def main():
    ctx = Context(); cfg = ctx.cfg
    seed_all(cfg["seed"])
    print(f"Runoff source: {ctx.run['source']} ({ctx.run['resolution']})")
    preds = pd.DataFrame(index=ctx.idx); preds.index.name = "date"
    preds["obs"] = ctx.q if ctx.q is not None else np.nan
    metrics = {}

    swat = load_swat(ctx.meta, ctx.idx)
    if swat is not None:
        preds["SWAT"] = swat.values
        metrics["SWAT (user)"] = {k: score(swat.values, ctx.obs, m, ctx.P)
                                  for k, m in [("calib", ctx.m_cal), ("valid", ctx.m_val),
                                               ("test", ctx.m_test), ("extreme_2021", ctx.m_ext)]}

    print("[Stage I] physics calibration with PSO / GWO / SSA ...")
    s1 = stage1(ctx, cfg)
    for name in cfg["stage1"]["optimizers"]:
        runs = [r for r in s1["runs"] if r["optimizer"] == name]
        best = max(runs, key=lambda r: 0.5 * r["valid_NSE"] + 0.5 * r["valid_KGE"])
        q, st, res = physics_periods(ctx, best["params"])
        preds[f"{name}-LoessHydro"] = q
        metrics[f"{name}-LoessHydro"] = res
    q, st, _ = physics_periods(ctx, s1["best"]["best_params"])
    p("results/predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v[:, 0] for k, v in st.items()} | {"Q": q}, index=ctx.idx).to_csv(
        p("results/predictions/physics_best_states.csv"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ctx.q is not None:
        bl = budget(cfg, "lstm")
        cal_idx, val_idx = np.flatnonzero(ctx.m_cal), np.flatnonzero(ctx.m_val)
        for cell in ["LSTM", "GRU"]:
            t0 = time.time()
            model, hist = train_seq2one(ctx.Xs, ctx.q, cal_idx, val_idx, cell, cfg["lstm"], bl, cfg["seed"], device)
            q = predict_seq2one(model, ctx.Xs, cfg["lstm"]["seq_len"], device)
            preds[cell] = q
            metrics[cell] = {k: score(np.nan_to_num(q, nan=np.nanmean(q)), ctx.obs, m, ctx.P)
                             for k, m in [("calib", ctx.m_cal), ("valid", ctx.m_val),
                                          ("test", ctx.m_test), ("extreme_2021", ctx.m_ext)]}
            torch.save(model.state_dict(), p(f"results/checkpoints/{cell}.pt"))
            print(f"  {cell} ({device}): valid NSE={metrics[cell]['valid']['NSE']:.3f}  "
                  f"test NSE={metrics[cell]['test']['NSE']:.3f}  ({time.time() - t0:.0f}s)")
    else:
        print("  monthly observations: LSTM/GRU seq2one baselines skipped (need daily targets)")

    print("[Base-PINN] default weights, no outer-loop optimisation ...")
    hb = budget(cfg, "hybrid")
    hp = dict(cfg["hybrid"]["default"])
    r = run_parallel([{"kind": "final", "hp": hp, "seed": cfg["seed"], "epochs": hb["epochs"],
                       "patience": hb["patience"], "save_path": "results/checkpoints/BasePINN.pt"}])[0]
    preds["Base-PINN"] = r["pred"]["Q"]
    metrics["Base-PINN"] = r["metrics"]
    save_json({"history": r["history"], "phys_params": r["phys_params"], "hp": hp},
              "results/metrics/basepinn_training.json")
    print(f"  Base-PINN: valid NSE={r['metrics']['valid']['NSE']:.3f} test NSE={r['metrics']['test']['NSE']:.3f}")

    preds.to_csv(p("results/predictions/baselines.csv"))
    save_json({"runoff_source": ctx.run["source"], "metrics": metrics}, "results/metrics/baselines_metrics.json")
    print("done.")


if __name__ == "__main__":
    main()
