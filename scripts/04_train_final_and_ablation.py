"""Phase 5b: final Meta-PINN-Hydro (multi-seed), ablation matrix and 50% small-sample stress test."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data.meteo_et0 import STORM_FEATURES  # noqa: E402
from src.models.baselines import predict_seq2one, train_seq2one  # noqa: E402
from src.models.pinn_hydro import score  # noqa: E402
from src.optimization.bilevel_tuner import Context  # noqa: E402
from src.optimization.jobs import run_parallel  # noqa: E402
from src.optimization.metaheuristics import minimize  # noqa: E402
from src.physics.loess_hydro_np import PARAM_NAMES, bounds_array, run_np, vec_to_params  # noqa: E402
from src.utils import budget, load_cfg, load_json, p, save_json  # noqa: E402

ABLATIONS = {
    "w/o Hard Physics": {"variant": "no_physics"},
    "w/o Soft Losses": {"variant": "no_soft"},
    "w/o Infiltration-Excess": {"variant": "no_inf"},
    "w/o Dynamic Modulation": {"variant": "no_modulation"},
    "w/o Storm Heterogeneity": {"drop_features": STORM_FEATURES},
}


def periods(ctx):
    return [("calib", ctx.m_cal), ("valid", ctx.m_val), ("test", ctx.m_test), ("extreme_2021", ctx.m_ext)]


def main():
    cfg = load_cfg()
    hb = budget(cfg, "hybrid")
    hp = load_json("results/checkpoints/best_meta_pinn_config.json")["hp"]
    seeds = cfg["final_seeds"]
    small = cfg["split"]["small_sample_calib"]
    base = {"kind": "final", "hp": hp, "epochs": hb["epochs"], "patience": hb["patience"]}

    jobs, tags = [], []
    for s in seeds:
        jobs.append(dict(base, seed=s, save_path=f"results/checkpoints/MetaPINN_seed{s}.pt")); tags.append(("seed", s))
    for name, extra in ABLATIONS.items():
        jobs.append(dict(base, seed=seeds[0], **extra)); tags.append(("ablation", name))
    jobs.append(dict(base, seed=seeds[0], calib_period=small)); tags.append(("small", "Meta-PINN-Hydro"))
    t0 = time.time()
    print(f"running {len(jobs)} hybrid fits in parallel ...")
    res = run_parallel(jobs)
    print(f"  done in {time.time() - t0:.0f}s")

    ctx = Context()
    preds = pd.DataFrame(index=ctx.idx); preds.index.name = "date"
    out = {"hp": hp, "seeds": {}, "ablation": {}, "small_sample": {}}
    Qs = []
    for (kind, tag), r in zip(tags, res):
        if kind == "seed":
            out["seeds"][str(tag)] = {"metrics": r["metrics"], "phys_params": r["phys_params"],
                                      "best_val_J": r["best_val_J"], "history": r["history"]}
            Qs.append(np.array(r["pred"]["Q"]))
            if tag == seeds[0]:
                st = pd.DataFrame({k: v for k, v in r["pred"].items()}, index=ctx.idx)
                st.to_csv(p("results/predictions/metapinn_states_seed0.csv"))
        elif kind == "ablation":
            out["ablation"][tag] = r["metrics"]; preds[f"abl:{tag}"] = r["pred"]["Q"]
        else:
            out["small_sample"][tag] = r["metrics"]
    Qs = np.stack(Qs)
    preds["Meta-PINN-Hydro"] = Qs.mean(0)
    preds["Meta-PINN-Hydro_p05"] = np.percentile(Qs, 5, axis=0)
    preds["Meta-PINN-Hydro_p95"] = np.percentile(Qs, 95, axis=0)
    for i, s in enumerate(seeds):
        preds[f"seed{s}"] = Qs[i]
    # ensemble metrics; WBE averaged over seeds (each seed's own physics closure)
    ens = {}
    for name, m in periods(ctx):
        wbes = [out["seeds"][str(s)]["metrics"][name]["WBE"] for s in seeds]
        ens[name] = score(preds["Meta-PINN-Hydro"].values, ctx.obs, m, ctx.P, wbe=float(np.mean(wbes)))
        obs = ctx.q[m] if ctx.q is not None else None
        if obs is not None:
            inside = (obs >= preds["Meta-PINN-Hydro_p05"].values[m]) & (obs <= preds["Meta-PINN-Hydro_p95"].values[m])
            ens[name]["band_coverage_pct"] = float(np.nanmean(inside) * 100)
    out["ensemble"] = ens
    per_seed = {name: {k: [out["seeds"][str(s)]["metrics"][name][k] for s in seeds] for k in ["NSE", "KGE"]}
                for name, _ in periods(ctx)}
    out["per_seed_summary"] = {n: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in d.items()}
                               for n, d in per_seed.items()}

    # small-sample comparators: conceptual (PSO) and LSTM calibrated on 2018-2019 only
    sctx = Context(calib_period=small)
    n = sctx.n_train
    lo, hi = bounds_array(cfg["physics"]["bounds"])
    from importlib import import_module
    sys.path.insert(0, str(p("scripts")))
    bs = import_module("02_train_baselines").batch_scores

    def obj(X):
        Q = run_np(sctx.P[:n], sctx.T[:n], sctx.E[:n], vec_to_params(X), sctx.CN0)
        nse, kge = bs(Q, sctx, sctx.m_cal)
        return -(0.5 * nse + 0.5 * kge)
    b1 = budget(cfg, "stage1")
    r = minimize("PSO", obj, lo, hi, pop=b1["pop"], iters=b1["iters"], seed=7)
    prm = dict(zip(PARAM_NAMES, map(float, r["x"])))
    q = run_np(sctx.P, sctx.T, sctx.E, {k: np.array([v]) for k, v in prm.items()}, sctx.CN0)[:, 0]
    out["small_sample"]["PSO-LoessHydro"] = {k: score(q, sctx.obs, m, sctx.P) for k, m in periods(sctx)}
    if sctx.q is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = train_seq2one(sctx.Xs, sctx.q, np.flatnonzero(sctx.m_cal), np.flatnonzero(sctx.m_val),
                                 "LSTM", cfg["lstm"], budget(cfg, "lstm"), cfg["seed"], device)
        ql = predict_seq2one(model, sctx.Xs, cfg["lstm"]["seq_len"], device)
        ql = np.nan_to_num(ql, nan=np.nanmean(ql))
        out["small_sample"]["LSTM"] = {k: score(ql, sctx.obs, m, sctx.P) for k, m in periods(sctx)}

    preds.to_csv(p("results/predictions/metapinn_final.csv"))
    save_json(out, "results/metrics/final_ablation_smallsample.json")
    print("Meta-PINN-Hydro ensemble test:", {k: round(v, 3) for k, v in ens["test"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
