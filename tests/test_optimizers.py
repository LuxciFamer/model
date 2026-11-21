import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.optimization.metaheuristics import minimize  # noqa: E402
from src.optimization.nsga2_optimizer import NSGA2, fast_non_dominated_sort, knee_point  # noqa: E402


def sphere(X):
    return np.sum((X - 0.3) ** 2, axis=1)


def rastrigin(X):
    return 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def test_single_objective_sphere():
    lo, hi = -5 * np.ones(5), 5 * np.ones(5)
    for name in ["PSO", "GWO", "SSA"]:
        r = minimize(name, sphere, lo, hi, pop=30, iters=150, seed=0)
        assert r["f"] < 1e-2, (name, r["f"])
        assert np.all(r["x"] >= lo) and np.all(r["x"] <= hi)
        h = r["history"]["best"]
        assert all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1))


def test_single_objective_rastrigin_beats_random():
    lo, hi = -5.12 * np.ones(5), 5.12 * np.ones(5)
    rng = np.random.default_rng(0)
    rand_best = rastrigin(rng.uniform(lo, hi, (30 * 150, 5))).min()
    for name in ["PSO", "GWO", "SSA"]:
        r = minimize(name, rastrigin, lo, hi, pop=30, iters=150, seed=1)
        assert r["f"] < rand_best, (name, r["f"], rand_best)


def zdt1(X):
    f1 = X[:, 0]
    g = 1 + 9 * X[:, 1:].mean(axis=1)
    return np.stack([f1, g * (1 - np.sqrt(f1 / g))], 1)


def test_nsga2_zdt1():
    d = 10
    r = NSGA2(zdt1, np.zeros(d), np.ones(d), pop=40, gens=120, seed=0).run()
    F = r["F"]
    assert len(F) >= 6
    fr = fast_non_dominated_sort(F)
    assert len(fr[0]) == len(F)  # returned set is mutually non-dominated
    gap = np.mean(F[:, 1] - (1 - np.sqrt(F[:, 0])))
    assert gap < 0.2, gap
    k = knee_point(F)
    assert 0 <= k < len(F)
