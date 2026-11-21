"""Phase 5a: outer loop II — PSO / GWO / SSA (single objective J) and NSGA-II (1-KGE vs |WBE|)
over physics-loss weights and network hyper-parameters. Validation period only."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.optimization.bilevel_tuner import J_COEF, SPACE_KEYS, decode, space_bounds  # noqa: E402
from src.optimization.jobs import batch_objective_factory, n_workers  # noqa: E402
from src.optimization.metaheuristics import minimize  # noqa: E402
from src.optimization.nsga2_optimizer import NSGA2, knee_point  # noqa: E402
from src.utils import budget, load_cfg, save_json  # noqa: E402


def main():
    cfg = load_cfg()
    b2, hb = budget(cfg, "stage2"), budget(cfg, "hybrid")
    lo, hi = space_bounds(cfg)
    base_job = {"kind": "val_eval", "seed": cfg["seed"], "epochs": hb["tune_epochs"],
                "patience": max(5, hb["tune_epochs"] // 3)}
    print(f"profile={cfg['profile']} workers={n_workers()} tune_epochs={hb['tune_epochs']}")
    out = {"space_keys": SPACE_KEYS, "lo": lo, "hi": hi, "single": {}, "j_coef": J_COEF}

    for name in ["PSO", "GWO", "SSA"]:
        t0 = time.time()
        cache = []
        f = batch_objective_factory(base_job, "single", cache, J_COEF)
        r = minimize(name, f, lo, hi, pop=b2["pop"], iters=b2["iters"], seed=11)
        out["single"][name] = {"best_x": r["x"], "best_hp": decode(r["x"]), "best_J": -r["f"],
                               "history": r["history"], "evals": cache, "seconds": time.time() - t0}
        print(f"  {name}: best val J={-r['f']:.4f}  hp={decode(r['x'])}  ({time.time() - t0:.0f}s)")

    t0 = time.time()
    cache = []
    f = batch_objective_factory(base_job, "multi", cache, J_COEF)
    nr = NSGA2(f, lo, hi, pop=b2["nsga_pop"], gens=b2["nsga_gens"], seed=13).run()
    k = knee_point(nr["F"])
    out["nsga2"] = {"pareto_X": nr["X"], "pareto_F": nr["F"], "knee_index": k,
                    "knee_hp": decode(nr["X"][k]), "knee_F": nr["F"][k], "history": nr["history"],
                    "evals": cache, "seconds": time.time() - t0}
    print(f"  NSGA-II: {len(nr['F'])} Pareto solutions; knee 1-KGE={nr['F'][k][0]:.4f} "
          f"|WBE|={nr['F'][k][1]:.4f}%  ({time.time() - t0:.0f}s)")

    # final configuration = NSGA-II knee point (accuracy / mass-balance trade-off)
    best = {"source": "NSGA-II knee", "hp": decode(nr["X"][k])}
    save_json(out, "results/metrics/stage2_outer_loop.json")
    save_json(best, "results/checkpoints/best_meta_pinn_config.json")
    print("best config:", best)


if __name__ == "__main__":
    main()
