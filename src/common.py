"""
Shared helpers for CS 422 Assignment 4.

Graphs are represented as an n x n integer matrix X, where X[i][j] is the number
of unit-capacity directed links from node i to node j (0 on the diagonal).
Traffic matrices T are n x n non-negative float matrices with zero diagonal.
"""
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------
def moore_avg_distance(n: int, d: int) -> float:
    """Lower bound on the average shortest-path length of ANY digraph with
    n nodes and out-degree d (directed Moore bound).

    From a fixed source, at most d nodes are at distance 1, d^2 at distance 2,
    ..., so the best possible distance profile fills those "shells" greedily.
    """
    remaining, k, total = n - 1, 1, 0
    while remaining > 0:
        count = min(d ** k, remaining)
        total += k * count
        remaining -= count
        k += 1
    return total / (n - 1)


def distance_matrix(X: np.ndarray) -> np.ndarray:
    """All-pairs hop distances via BFS (np.inf if unreachable)."""
    n = X.shape[0]
    adj = [[j for j in range(n) if X[i, j] > 0] for i in range(n)]
    D = np.full((n, n), np.inf)
    for s in range(n):
        D[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if D[s, v] == np.inf:
                    D[s, v] = D[s, u] + 1
                    q.append(v)
    return D


def avg_distance(X: np.ndarray) -> float:
    n = X.shape[0]
    D = distance_matrix(X)
    return D.sum() / (n * (n - 1))


def check_regular(X: np.ndarray, d: int) -> None:
    assert (np.diag(X) == 0).all(), "self-loops present"
    assert (X.sum(axis=1) == d).all(), f"out-degrees {X.sum(axis=1)} != {d}"
    assert (X.sum(axis=0) == d).all(), f"in-degrees {X.sum(axis=0)} != {d}"


# ---------------------------------------------------------------------------
# Maximum concurrent flow on a FIXED topology (LP)
# ---------------------------------------------------------------------------
def max_concurrent_flow(X: np.ndarray, T: np.ndarray, solver: str = "auto",
                        verbose: bool = False):
    """Max theta such that theta*T can be routed (splittable, multipath) on
    topology X with unit capacity per link.

    Source-aggregated multi-commodity formulation: f[s, (i,j)] is the flow on
    link (i,j) originating at source s.

    Returns (theta, link_load_matrix).
    solver: "gurobi", "scipy", or "auto" (gurobi, falling back to scipy/HiGHS
    if the license is too small for the instance).
    """
    if solver in ("auto", "gurobi"):
        try:
            return _mcf_gurobi(X, T, verbose)
        except Exception as e:  # e.g. size-limited pip license
            if solver == "gurobi":
                raise
            if verbose:
                print(f"[gurobi unavailable for this size: {e}] -> scipy")
    return _mcf_scipy(X, T)


def _mcf_gurobi(X, T, verbose=False):
    import gurobipy as gp
    from gurobipy import GRB

    n = X.shape[0]
    E = [(i, j) for i in range(n) for j in range(n) if X[i, j] > 0]
    S = [s for s in range(n) if T[s].sum() > 0]
    m = gp.Model("mcf")
    m.Params.OutputFlag = 1 if verbose else 0
    theta = m.addVar(lb=0.0, name="theta")
    f = m.addVars(S, E, lb=0.0, name="f")
    for s in S:
        for v in range(n):
            out_f = gp.quicksum(f[s, v, j] for j in range(n) if X[v, j] > 0)
            in_f = gp.quicksum(f[s, i, v] for i in range(n) if X[i, v] > 0)
            if v == s:
                m.addConstr(out_f - in_f == theta * T[s].sum())
            else:
                m.addConstr(in_f - out_f == theta * T[s, v])
    for (i, j) in E:
        m.addConstr(gp.quicksum(f[s, i, j] for s in S) <= X[i, j])
    m.setObjective(theta, GRB.MAXIMIZE)
    m.optimize()
    load = np.zeros((n, n))
    for (i, j) in E:
        load[i, j] = sum(f[s, i, j].X for s in S)
    return theta.X, load


def _mcf_scipy(X, T):
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n = X.shape[0]
    E = [(i, j) for i in range(n) for j in range(n) if X[i, j] > 0]
    S = [s for s in range(n) if T[s].sum() > 0]
    eidx = {e: k for k, e in enumerate(E)}
    nv = 1 + len(S) * len(E)          # var 0 = theta
    var = lambda si, e: 1 + si * len(E) + eidx[e]

    Aeq = lil_matrix((len(S) * n, nv)); beq = np.zeros(len(S) * n)
    r = 0
    for si, s in enumerate(S):
        for v in range(n):
            sign = 1 if v == s else -1          # out-in (=theta*.)  for s
            for j in range(n):
                if X[v, j] > 0:
                    Aeq[r, var(si, (v, j))] += sign
            for i in range(n):
                if X[i, v] > 0:
                    Aeq[r, var(si, (i, v))] -= sign
            Aeq[r, 0] = -(T[s].sum() if v == s else T[s, v])
            r += 1
    Aub = lil_matrix((len(E), nv)); bub = np.zeros(len(E))
    for k, (i, j) in enumerate(E):
        for si in range(len(S)):
            Aub[k, var(si, (i, j))] = 1
        bub[k] = X[i, j]
    c = np.zeros(nv); c[0] = -1
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    load = np.zeros((n, n))
    for (i, j) in E:
        load[i, j] = sum(res.x[var(si, (i, j))] for si in range(len(S)))
    return res.x[0], load


def uniform_tm(n: int, d: int) -> np.ndarray:
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0)
    return T
