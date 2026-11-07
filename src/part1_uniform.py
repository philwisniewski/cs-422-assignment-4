"""
Part 1 -- Best topology for the UNIFORM traffic matrix.

    T[i][j] = d/(n-1)  for all i != j,    n nodes, d unit links in / out per node.

Key fact (see report): for the uniform TM
    theta*(G) <= 1 / avg_shortest_path(G) <= (n-1) / M(n,d)
where M(n,d) is the directed Moore bound on the total distance from one node.
So the best topology is the d-regular digraph with the smallest average
shortest-path length whose shortest paths can be loaded evenly.

Construction (build_topology) -- several candidates, keep the best theta:
  A. d == n-1 -> complete digraph (every pair 1 hop, theta = 1).
  B. "Cyclotomic" Cayley digraph (uses d | n-1; needs n prime):
         i -> i + h (mod n)   for every h in H,
     H = the unique multiplicative subgroup of Z_n^* of order d.
     x -> h*x and x -> x+c are automorphisms, so the graph is ARC-TRANSITIVE:
     every link looks the same, the optimal routing loads all links equally
     and theta = 1/avg_dist EXACTLY. Moore-optimal e.g. for (7,3),(13,4),(11,5).
  C. Generalized Kautz / Imase-Itoh digraph  i -> (-d*i - a) mod n, a=1..d
     (exists for all n,d; diameter <= ceil(log_d n); exactly Moore-optimal at
     Kautz sizes), self-loops removed, then a degree-preserving local search
     that lowers the average shortest-path length.
  D. Keep the best candidate by LP theta and refine it with a local search
     whose objective is the LP theta itself (stops if theta hits the bound).

Usage:
    python src/part1_uniform.py                 # default sweep of (n, d)
    python src/part1_uniform.py --n 13 --d 4    # one instance + figure
"""
import argparse
import json
import os
import random

import numpy as np

from common import (avg_distance, check_regular, distance_matrix,
                    max_concurrent_flow, moore_avg_distance, uniform_tm)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


# ---------------------------------------------------------------------------
# Step 2: generalized Kautz (Imase-Itoh) digraph
# ---------------------------------------------------------------------------
def imase_itoh(n: int, d: int) -> np.ndarray:
    X = np.zeros((n, n), dtype=int)
    for i in range(n):
        for a in range(1, d + 1):
            X[i, (-d * i - a) % n] += 1
    return X


# ---------------------------------------------------------------------------
# Step 3: self-loop repair (keeps every in/out degree equal to d)
# ---------------------------------------------------------------------------
def remove_self_loops(X: np.ndarray, rng: random.Random) -> np.ndarray:
    X = X.copy()
    n = X.shape[0]
    while (np.diag(X) > 0).any():
        u = int(np.flatnonzero(np.diag(X))[0])
        edges = [(c, e) for c in range(n) for e in range(n)
                 if X[c, e] > 0 and c != u and e != u
                 and X[u, e] == 0 and X[c, u] == 0]
        c, e = rng.choice(edges)
        # u->u, c->e   becomes   u->e, c->u
        X[u, u] -= 1; X[c, e] -= 1
        X[u, e] += 1; X[c, u] += 1
    return X


# ---------------------------------------------------------------------------
# Step 4: degree-preserving local search on the average path length
# ---------------------------------------------------------------------------
def local_search(X: np.ndarray, d: int, iters: int = 20000, seed: int = 0):
    rng = random.Random(seed)
    n = X.shape[0]
    target = moore_avg_distance(n, d)
    best = avg_distance(X)
    for _ in range(iters):
        if best <= target + 1e-12:
            break                                    # provably optimal
        edges = [(i, j) for i in range(n) for j in range(n) if X[i, j] > 0]
        (a, b), (c, e) = rng.sample(edges, 2)
        # swap heads: a->b, c->e   becomes   a->e, c->b
        if a == e or c == b or X[a, e] > 0 or X[c, b] > 0:
            continue
        X[a, b] -= 1; X[c, e] -= 1; X[a, e] += 1; X[c, b] += 1
        val = avg_distance(X)
        if val <= best:            # accept improving and sideways moves
            best = val
        else:
            X[a, b] += 1; X[c, e] += 1; X[a, e] -= 1; X[c, b] -= 1
    return X, best


# ---------------------------------------------------------------------------
# Candidate B: cyclotomic Cayley digraph on Z_n (n prime, d | n-1)
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    return n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1))


def order_d_subgroup(n: int, d: int):
    """The unique subgroup H of the cyclic group Z_n^* with |H| = d."""
    g = next(g for g in range(2, n)
             if len({pow(g, k, n) for k in range(1, n)}) == n - 1)  # generator
    h = pow(g, (n - 1) // d, n)
    return sorted({pow(h, k, n) for k in range(d)})


def cyclotomic(n: int, d: int):
    if not (is_prime(n) and (n - 1) % d == 0):
        return None
    H = order_d_subgroup(n, d)
    X = np.zeros((n, n), dtype=int)
    for i in range(n):
        for h in H:
            X[i, (i + h) % n] = 1
    return X


# ---------------------------------------------------------------------------
# Step D: local search that maximises the LP theta directly
# ---------------------------------------------------------------------------
def theta_search(X, d, theta, iters, seed, solver):
    rng = random.Random(seed)
    n = X.shape[0]
    T = uniform_tm(n, d)
    ub = 1 / moore_avg_distance(n, d)
    for _ in range(iters):
        if theta >= ub - 1e-9:
            break                                    # provably optimal
        edges = [(i, j) for i in range(n) for j in range(n) if X[i, j] > 0]
        (a, b), (c, e) = rng.sample(edges, 2)
        if a == e or c == b or X[a, e] > 0 or X[c, b] > 0:
            continue
        X[a, b] -= 1; X[c, e] -= 1; X[a, e] += 1; X[c, b] += 1
        t, _ = max_concurrent_flow(X, T, solver=solver)
        if t >= theta - 1e-12:
            theta = t
        else:
            X[a, b] += 1; X[c, e] += 1; X[a, e] -= 1; X[c, b] -= 1
    return X, theta


def build_topology(n: int, d: int, seed: int = 0, solver: str = "auto",
                   theta_iters: int = 150, verbose: bool = False):
    """Returns (X, name_of_winning_construction)."""
    assert 1 <= d <= n - 1
    if d == n - 1:
        X = np.ones((n, n), dtype=int)
        np.fill_diagonal(X, 0)
        return X, "complete"
    T = uniform_tm(n, d)
    rng = random.Random(seed)
    cands = {}
    Xc = cyclotomic(n, d)
    if Xc is not None:
        cands["cyclotomic"] = Xc
    Xk = remove_self_loops(imase_itoh(n, d), rng)
    cands["kautz+search"], _ = local_search(Xk, d, seed=seed)
    scored = {k: max_concurrent_flow(v, T, solver=solver)[0]
              for k, v in cands.items()}
    if verbose:
        print("    candidates:", {k: round(v, 4) for k, v in scored.items()})
    name = max(scored, key=scored.get)
    X, theta = cands[name].copy(), scored[name]
    X, theta2 = theta_search(X, d, theta, theta_iters, seed, solver)
    if theta2 > theta + 1e-9:
        name += " + theta-search"
    check_regular(X, d)
    return X, name


def analyze(n: int, d: int, seed: int = 0, solver: str = "auto") -> dict:
    X, name = build_topology(n, d, seed, solver)
    L = avg_distance(X)
    M = moore_avg_distance(n, d)
    theta, load = max_concurrent_flow(X, uniform_tm(n, d), solver=solver)
    return dict(n=n, d=d, construction=name,
                moore_avg_dist=M, theta_upper_bound=1 / M,
                avg_dist=L, one_over_avg_dist=1 / L,
                theta_lp=theta,
                gap_to_bound_pct=100 * (1 - theta * M),
                kautz_avg_dist=avg_distance(remove_self_loops(
                    imase_itoh(n, d), random.Random(seed))) if d < n - 1 else 1.0,
                max_link_util=float(load.max()) / theta if theta > 0 else 0,
                edges=[[int(i), int(j)] for i, j in zip(*np.nonzero(X))])


def draw(X: np.ndarray, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.DiGraph()
    n = X.shape[0]
    G.add_nodes_from(range(n))
    G.add_edges_from((i, j) for i in range(n) for j in range(n) if X[i, j])
    plt.figure(figsize=(6, 6))
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, node_color="#cfe3f7", edgecolors="#2b5d8a",
                     node_size=500, arrowsize=10, width=0.8,
                     connectionstyle="arc3,rad=0.08")
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--d", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default="auto", choices=["auto", "gurobi", "scipy"])
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    if args.n:
        cases = [(args.n, args.d)]
    else:  # sweep; n-1 divisible by d as in the assignment
        cases = [(5, 2), (7, 2), (7, 3), (9, 2), (9, 4), (10, 3), (13, 3),
                 (13, 4), (13, 6), (16, 3), (17, 4), (19, 6), (21, 4), (21, 5)]

    rows = []
    hdr = f"{'n':>3} {'d':>2} | {'Moore L':>8} {'Kautz L':>8} {'ours L':>8} | " \
          f"{'theta UB':>8} {'theta LP':>8} {'gap%':>6}  winner"
    print(hdr); print("-" * len(hdr))
    for n, d in cases:
        r = analyze(n, d, args.seed, args.solver)
        rows.append(r)
        print(f"{n:>3} {d:>2} | {r['moore_avg_dist']:8.4f} {r['kautz_avg_dist']:8.4f} "
              f"{r['avg_dist']:8.4f} | {r['theta_upper_bound']:8.4f} "
              f"{r['theta_lp']:8.4f} {r['gap_to_bound_pct']:6.2f}  {r['construction']}")
        if len(cases) == 1 or (n, d) == (13, 4):
            X = np.zeros((n, n), dtype=int)
            for i, j in r["edges"]:
                X[i, j] = 1
            fig = os.path.join(RESULTS, f"part1_topology_n{n}_d{d}.png")
            draw(X, fig, f"Part 1 topology, n={n}, d={d}, "
                         f"theta={r['theta_lp']:.3f}")
            D = distance_matrix(X)
            prof = {int(k): int((D[0] == k).sum()) for k in np.unique(D[0]) if k > 0}
            print(f"    distance profile from node 0: {prof}   figure: {fig}")

    out = os.path.join(RESULTS, f"part1_n{args.n}_d{args.d}.json" if args.n
                       else "part1_results.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
