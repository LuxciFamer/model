"""Process-pool job runner for hybrid-model fits (outer loop II, final seeds, ablations).

Each worker builds its own Context (cached per calibration period / feature set) and
runs single-threaded PyTorch, so candidates are trained in parallel across CPU cores.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_CTX = {}


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _ctx(calib_period, drop_features):
    from src.data.meteo_et0 import FEATURES_ALL
    from src.optimization.bilevel_tuner import Context
    key = (tuple(calib_period) if calib_period else None, tuple(drop_features or ()))
    if key not in _CTX:
        feats = [f for f in FEATURES_ALL if f not in (drop_features or ())]
        _CTX[key] = Context(calib_period=calib_period, features=feats)
    return _CTX[key]


def run_job(job: dict) -> dict:
    import torch
    from src.optimization.bilevel_tuner import evaluate_periods, fit, predict_full, val_eval
    from src.utils import load_json
    priors = load_json("results/checkpoints/stage1_physics_priors.json")["best_params"]
    ctx = _ctx(job.get("calib_period"), job.get("drop_features"))
    if job["kind"] == "val_eval":
        try:
            m = val_eval(ctx, job["hp"], priors, job["epochs"], job["patience"], job["seed"])
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": repr(e)}
        return {"ok": True, "metrics": m}
    model, hist, best = fit(ctx, job["hp"], priors, job["epochs"], job["patience"], job["seed"],
                            variant=job.get("variant", "full"))
    out = predict_full(ctx, model)
    res = evaluate_periods(ctx, out["Q"], out)
    if job.get("save_path"):
        from src.utils import p
        path = p(job["save_path"]); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    phys = model.phys.params_numpy() if hasattr(model, "phys") else None
    return {"ok": True, "metrics": res, "history": hist, "best_val_J": best,
            "pred": {k: v.tolist() for k, v in out.items() if k != "P"}, "phys_params": phys}


def n_workers():
    return max(1, min(int(os.environ.get("XS_WORKERS", 0)) or (os.cpu_count() or 2) - 2, 6))


def run_parallel(jobs: list[dict]) -> list[dict]:
    if len(jobs) == 1 or n_workers() == 1:
        _init_worker()
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(n_workers(), len(jobs)), initializer=_init_worker) as ex:
        return list(ex.map(run_job, jobs))


def batch_objective_factory(base_job: dict, mode: str, cache: list, j_coef: float):
    """Return func(X)->costs for outer-loop optimisers; logs every evaluation into `cache`."""
    from src.optimization.bilevel_tuner import decode

    def func(X):
        jobs = [dict(base_job, hp=decode(x)) for x in X]
        res = run_parallel(jobs)
        costs = []
        for x, r in zip(X, res):
            if not r["ok"]:
                m = {"KGE": -10.0, "NSE": -10.0, "WBE": 100.0}
            else:
                m = r["metrics"]
            wbe = abs(m["WBE"]) if np.isfinite(m["WBE"]) else 100.0
            J = 0.5 * m["KGE"] + 0.5 * m["NSE"] - j_coef * wbe
            cache.append({"x": list(map(float, x)), "hp": decode(x), "KGE": m["KGE"],
                          "NSE": m["NSE"], "WBE": m["WBE"], "J": J})
            costs.append(-J if mode == "single" else [1 - m["KGE"], wbe])
        return np.array(costs)

    return func
