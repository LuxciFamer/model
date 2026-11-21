"""NSGA-II (Deb et al., 2002) for bi/multi-objective minimisation.

`func` maps an (n, d) array in REAL space to an (n, m) objective array.
"""
from __future__ import annotations

import numpy as np


def fast_non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    n = len(F)
    S = [[] for _ in range(n)]
    nd = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        idx = np.argsort(F[:, j])
        d[idx[0]] = d[idx[-1]] = np.inf
        rng = F[idx[-1], j] - F[idx[0], j]
        if rng <= 0:
            continue
        d[idx[1:-1]] += (F[idx[2:], j] - F[idx[:-2], j]) / rng
    return d


class NSGA2:
    name = "NSGA-II"

    def __init__(self, func, lo, hi, pop=40, gens=50, seed=0, eta_c=15, eta_m=20, pc=0.9):
        self.func = func
        self.lo = np.asarray(lo, float); self.hi = np.asarray(hi, float)
        self.d = len(self.lo); self.pop = pop + (pop % 2); self.gens = gens
        self.rng = np.random.default_rng(seed)
        self.eta_c, self.eta_m, self.pc = eta_c, eta_m, pc
        self.pm = 1.0 / self.d
        self.history = []
        self.archive_X, self.archive_F = [], []

    def _u2x(self, U):
        return self.lo + U * (self.hi - self.lo)

    def _eval(self, U):
        F = np.asarray(self.func(self._u2x(U)), float)
        F = np.where(np.isfinite(F), F, 1e6)
        self.archive_X.append(self._u2x(U)); self.archive_F.append(F)
        return F

    def _tournament(self, rank, crowd):
        a, b = self.rng.integers(0, len(rank), 2)
        if rank[a] < rank[b] or (rank[a] == rank[b] and crowd[a] > crowd[b]):
            return a
        return b

    def _sbx(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if self.rng.random() > self.pc:
            return c1, c2
        for i in range(self.d):
            if self.rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-14:
                continue
            u = self.rng.random()
            beta = (2 * u) ** (1 / (self.eta_c + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def _mutate(self, c):
        for i in range(self.d):
            if self.rng.random() < self.pm:
                u = self.rng.random()
                delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
                c[i] = c[i] + delta
        return np.clip(c, 0, 1)

    @staticmethod
    def _rank_crowd(F):
        fronts = fast_non_dominated_sort(F)
        rank = np.zeros(len(F), int); crowd = np.zeros(len(F))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding_distance(F[fr])
        return fronts, rank, crowd

    def run(self):
        U = self.rng.random((self.pop, self.d))
        F = self._eval(U)
        fronts, rank, crowd = self._rank_crowd(F)
        for g in range(self.gens):
            kids = []
            while len(kids) < self.pop:
                a = U[self._tournament(rank, crowd)]; b = U[self._tournament(rank, crowd)]
                c1, c2 = self._sbx(a, b)
                kids += [self._mutate(c1), self._mutate(c2)]
            K = np.array(kids[:self.pop])
            FK = self._eval(K)
            UU = np.vstack([U, K]); FF = np.vstack([F, FK])
            fronts, _, _ = self._rank_crowd(FF)
            sel = []
            for fr in fronts:
                if len(sel) + len(fr) <= self.pop:
                    sel += fr
                else:
                    cd = crowding_distance(FF[fr])
                    sel += [fr[i] for i in np.argsort(-cd)[: self.pop - len(sel)]]
                    break
            U, F = UU[sel], FF[sel]
            fronts, rank, crowd = self._rank_crowd(F)
            self.history.append({"gen": g + 1, "front_size": len(fronts[0]),
                                 "min_f": F.min(axis=0).tolist()})
        pf = fronts[0]
        return {"X": self._u2x(U[pf]), "F": F[pf], "pop_X": self._u2x(U), "pop_F": F,
                "history": self.history,
                "archive_X": np.vstack(self.archive_X), "archive_F": np.vstack(self.archive_F)}


def knee_point(F: np.ndarray) -> int:
    """Index of the Pareto point closest (normalised) to the ideal point."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    Z = (F - lo) / np.where(hi - lo > 0, hi - lo, 1)
    return int(np.argmin(np.linalg.norm(Z, axis=1)))
