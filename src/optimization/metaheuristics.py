"""Population-based optimisers: PSO, GWO, SSA (minimisation).

All optimisers search in a normalised box. `func` receives an (n, d) array in the
REAL parameter space and returns an (n,) array of costs (lower is better).
History (best / mean cost and population diversity per iteration) is recorded.
"""
from __future__ import annotations

import numpy as np


class _Base:
    name = "base"

    def __init__(self, func, lo, hi, pop=30, iters=80, seed=0):
        self.func = func
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.d = len(self.lo); self.pop = pop; self.iters = iters
        self.rng = np.random.default_rng(seed)
        self.history = {"best": [], "mean": [], "diversity": []}
        self.n_evals = 0

    # normalised coordinates z in [-1, 1]
    def _real(self, z):
        return self.lo + (np.clip(z, -1, 1) + 1) / 2 * (self.hi - self.lo)

    def _eval(self, z):
        f = np.asarray(self.func(self._real(z)), float)
        f = np.where(np.isfinite(f), f, 1e12)
        self.n_evals += len(z)
        return f

    def _log(self, z, f):
        self.history["best"].append(float(self.best_f))
        self.history["mean"].append(float(np.mean(f)))
        self.history["diversity"].append(float(np.mean(np.std(z, axis=0))))

    def _update_best(self, z, f):
        i = int(np.argmin(f))
        if f[i] < self.best_f:
            self.best_f = float(f[i]); self.best_z = z[i].copy()

    def result(self):
        return {"x": self._real(self.best_z), "f": self.best_f, "history": self.history,
                "n_evals": self.n_evals, "name": self.name}


class PSO(_Base):
    """Particle swarm (Kennedy & Eberhart, 1995) with linearly decreasing inertia."""
    name = "PSO"

    def run(self, c1=1.5, c2=1.5, w0=0.9, w1=0.4, vmax=0.4):
        z = self.rng.uniform(-1, 1, (self.pop, self.d))
        v = self.rng.uniform(-vmax, vmax, (self.pop, self.d))
        f = self._eval(z)
        pb, pf = z.copy(), f.copy()
        self.best_f, self.best_z = np.inf, None
        self._update_best(z, f); self._log(z, f)
        for t in range(1, self.iters):
            w = w0 - (w0 - w1) * t / max(1, self.iters - 1)
            r1, r2 = self.rng.random(z.shape), self.rng.random(z.shape)
            v = w * v + c1 * r1 * (pb - z) + c2 * r2 * (self.best_z - z)
            v = np.clip(v, -vmax, vmax)
            z = np.clip(z + v, -1, 1)
            f = self._eval(z)
            imp = f < pf
            pb[imp], pf[imp] = z[imp], f[imp]
            self._update_best(z, f); self._log(z, f)
        return self.result()


class GWO(_Base):
    """Grey Wolf Optimizer (Mirjalili et al., 2014)."""
    name = "GWO"

    def run(self):
        z = self.rng.uniform(-1, 1, (self.pop, self.d))
        f = self._eval(z)
        self.best_f, self.best_z = np.inf, None
        self._update_best(z, f); self._log(z, f)
        for t in range(1, self.iters):
            idx = np.argsort(f)
            leaders = z[idx[:3]]
            a = 2 - 2 * t / max(1, self.iters - 1)
            new = np.zeros_like(z)
            for L in leaders:
                A = 2 * a * self.rng.random(z.shape) - a
                C = 2 * self.rng.random(z.shape)
                new += L - A * np.abs(C * L - z)
            z = np.clip(new / 3, -1, 1)
            f = self._eval(z)
            self._update_best(z, f); self._log(z, f)
        return self.result()


class SSA(_Base):
    """Sparrow Search Algorithm (Xue & Shen, 2020): producers / scroungers / scouts."""
    name = "SSA"

    def run(self, pd_ratio=0.2, sd_ratio=0.1, ST=0.8):
        n = self.pop
        z = self.rng.uniform(-1, 1, (n, self.d))
        f = self._eval(z)
        self.best_f, self.best_z = np.inf, None
        self._update_best(z, f); self._log(z, f)
        n_pd = max(1, int(round(pd_ratio * n))); n_sd = max(1, int(round(sd_ratio * n)))
        for t in range(1, self.iters):
            idx = np.argsort(f); z, f = z[idx], f[idx]
            xbest, xworst, fbest, fworst = z[0].copy(), z[-1].copy(), f[0], f[-1]
            new = z.copy()
            R2 = self.rng.random()
            for i in range(n_pd):  # producers
                if R2 < ST:
                    alpha = self.rng.uniform(1e-3, 1)
                    new[i] = z[i] * np.exp(-(i + 1) / (alpha * self.iters))
                else:
                    new[i] = z[i] + self.rng.normal() * np.ones(self.d)
            xp = new[0].copy()
            for i in range(n_pd, n):  # scroungers
                if i > n / 2:
                    new[i] = self.rng.normal() * np.exp((xworst - z[i]) / (i + 1) ** 2)
                else:
                    A = self.rng.choice([-1.0, 1.0], self.d)
                    step = np.abs(z[i] - xp) @ (A / self.d)
                    new[i] = xp + step * np.ones(self.d)
            for i in self.rng.choice(n, n_sd, replace=False):  # scouts (danger aware)
                if f[i] > fbest:
                    new[i] = xbest + self.rng.normal() * np.abs(z[i] - xbest)
                else:
                    K = self.rng.uniform(-1, 1)
                    new[i] = z[i] + K * np.abs(z[i] - xworst) / (f[i] - fworst + 1e-12)
            z = np.clip(new, -1, 1)
            f = self._eval(z)
            self._update_best(z, f); self._log(z, f)
        return self.result()


OPTIMIZERS = {"PSO": PSO, "GWO": GWO, "SSA": SSA}


def minimize(name, func, lo, hi, pop=30, iters=80, seed=0):
    return OPTIMIZERS[name](func, lo, hi, pop=pop, iters=iters, seed=seed).run()
