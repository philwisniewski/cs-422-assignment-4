"""
Part 2 -- Traffic-aware topology design as a Mixed-Integer Linear Program
(Gurobi), n = 8 nodes, d = 4 unit-capacity links in and out of every node.

Given a traffic matrix T from the hose set, choose the links AND the routing
jointly so that the concurrent flow theta is maximised.

  Decision variables
    x[i,j]   integer >= 0   number of unit links i -> j     (i != j)
                            (binary if --simple: at most one link per pair)
    f[s,i,j] real    >= 0   flow on link i->j that originates at source s
    theta    real    >= 0   concurrent-flow scaling factor

  Objective      maximise theta

  Constraints
    (1) out-degree   sum_j x[i,j] = d                         for all i
    (2) in-degree    sum_i x[i,j] = d                         for all j
    (3) conservation for each source s and node v:
           v == s : sum_j f[s,s,j] - sum_i f[s,i,s] = theta * sum_t T[s,t]
           v != s : sum_i f[s,i,v] - sum_j f[s,v,j] = theta * T[s,v]
    (4) capacity     sum_s f[s,i,j] <= x[i,j]                  for all i != j
    (5) no flow back into its own source: f[s,i,s] = 0   (tightening only)

  Because the capacities are the integer variables x, solving the MILP picks
  the topology that is best FOR THIS T (the topology adapts to the TM). The
  per-source aggregation keeps the model small: n*n(n-1) + n(n-1) + 1 = 505
  variables for n = 8.

Optional --robust K: one topology for K sampled hose TMs at once
(maximise min_k theta_k), i.e. a static topology that is good for all of them.

Usage
    python src/part2_milp.py                    # run the experiment suite
    python src/part2_milp.py --tm my_tm.csv     # solve for your own 8x8 TM
    python src/part2_milp.py --simple           # forbid parallel links
    python src/part2_milp.py --robust 3         # one topology for 3 TMs
"""
import argparse
import json
import os
import sys

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import max_concurrent_flow, check_regular        # noqa: E402
from hose_tms import GENERATORS, is_hose                     # noqa: E402
from part1_uniform import build_topology                     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


# ===========================================================================
# THE MODEL
# ===========================================================================
def best_topology_milp(T, d=4, simple=False, time_limit=120, verbose=False):
    """Solve the joint topology + routing MILP for traffic matrix T.

    Returns dict(theta, X (n x n int), load (n x n), status, gap, runtime).
    """
    T = np.asarray(T, dtype=float)
    n = T.shape[0]
    N = range(n)
    P = [(i, j) for i in N for j in N if i != j]          # candidate links
    S = [s for s in N if T[s].sum() > 1e-12]              # active sources

    m = gp.Model("best_topology")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit

    # ---- decision variables ------------------------------------------------
    vtype = GRB.BINARY if simple else GRB.INTEGER
    x = m.addVars(P, vtype=vtype, lb=0, ub=1 if simple else d, name="x")
    f = m.addVars(S, P, lb=0.0, name="f")
    theta = m.addVar(lb=0.0, name="theta")

    # ---- objective -----------------------------------------------------------
    m.setObjective(theta, GRB.MAXIMIZE)

    # ---- (1),(2) degree constraints ----------------------------------------
    m.addConstrs((x.sum(i, "*") == d for i in N), name="out_deg")
    m.addConstrs((x.sum("*", j) == d for j in N), name="in_deg")

    # ---- (3) flow conservation ---------------------------------------------
    for s in S:
        for v in N:
            out_f = gp.quicksum(f[s, v, j] for j in N if j != v)
            in_f = gp.quicksum(f[s, i, v] for i in N if i != v)
            if v == s:
                m.addConstr(out_f - in_f == theta * T[s].sum(), name=f"src[{s}]")
            else:
                m.addConstr(in_f - out_f == theta * T[s, v], name=f"cons[{s},{v}]")

    # ---- (4) capacity: flow only on links that exist ------------------------
    m.addConstrs((f.sum("*", i, j) <= x[i, j] for (i, j) in P), name="cap")

    # ---- (5) tightening: no flow re-enters its source ---------------------
    for s in S:
        for i in N:
            if i != s:
                f[s, i, s].UB = 0.0

    m.optimize()
    return _extract(m, x, f, theta, n, S, P)


def robust_topology_milp(Ts, d=4, simple=False, time_limit=300, verbose=False):
    """One topology x shared by several TMs; separate routing per TM;
    maximise the worst theta. Same constraints as above, replicated per TM."""
    n = Ts[0].shape[0]
    N = range(n)
    P = [(i, j) for i in N for j in N if i != j]
    K = range(len(Ts))
    m = gp.Model("robust_topology")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    vtype = GRB.BINARY if simple else GRB.INTEGER
    x = m.addVars(P, vtype=vtype, lb=0, ub=1 if simple else d, name="x")
    f = m.addVars(K, N, P, lb=0.0, name="f")
    th = m.addVars(K, lb=0.0, name="theta_k")
    z = m.addVar(lb=0.0, name="z")                       # z = min_k theta_k
    m.setObjective(z, GRB.MAXIMIZE)
    m.addConstrs((z <= th[k] for k in K))
    m.addConstrs((x.sum(i, "*") == d for i in N))
    m.addConstrs((x.sum("*", j) == d for j in N))
    for k in K:
        T = Ts[k]
        for s in N:
            for v in N:
                out_f = gp.quicksum(f[k, s, v, j] for j in N if j != v)
                in_f = gp.quicksum(f[k, s, i, v] for i in N if i != v)
                rhs = th[k] * (T[s].sum() if v == s else T[s, v])
                m.addConstr((out_f - in_f if v == s else in_f - out_f) == rhs)
        m.addConstrs((gp.quicksum(f[k, s, i, j] for s in N) <= x[i, j]
                      for (i, j) in P))
    m.optimize()
    X = np.zeros((n, n), dtype=int)
    for (i, j) in P:
        X[i, j] = int(round(x[i, j].X))
    return dict(theta=z.X, X=X, thetas=[th[k].X for k in K],
                status=m.Status, gap=m.MIPGap, runtime=m.Runtime)


def _extract(m, x, f, theta, n, S, P):
    if m.SolCount == 0:
        raise RuntimeError(f"no solution found (status {m.Status})")
    X = np.zeros((n, n), dtype=int)
    load = np.zeros((n, n))
    for (i, j) in P:
        X[i, j] = int(round(x[i, j].X))
        load[i, j] = sum(f[s, i, j].X for s in S)
    return dict(theta=theta.X, X=X, load=load, status=m.Status,
                gap=m.MIPGap, runtime=m.Runtime)


# ===========================================================================
# Experiment driver
# ===========================================================================
def fmt(X):
    return "\n".join("      " + " ".join(str(v) for v in row) for row in X)


def run_suite(args):
    n, d = 8, 4
    rng = np.random.default_rng(args.seed)
    X_static, how = build_topology(n, d)       # Part-1 (uniform-optimal) graph
    rows = []
    print(f"Static baseline = Part-1 topology for uniform TM ({how})\n")
    print(f"{'traffic matrix':<16}{'theta MILP':>11}{'theta static':>13}"
          f"{'gain':>7}{'gap':>7}{'time[s]':>9}")
    print("-" * 63)
    for name, gen in GENERATORS.items():
        for rep in range(args.reps if name != "uniform" else 1):
            T = gen(n, d, rng)
            assert is_hose(T, d)
            r = best_topology_milp(T, d, args.simple, args.time_limit, args.verbose)
            check_regular(r["X"], d)
            t_static, _ = max_concurrent_flow(X_static, T, solver="gurobi")
            # sanity: re-check MILP topology with the independent LP
            t_check, _ = max_concurrent_flow(r["X"], T, solver="gurobi")
            assert abs(t_check - r["theta"]) < 1e-5, (t_check, r["theta"])
            label = name if args.reps == 1 or name == "uniform" else f"{name}#{rep}"
            print(f"{label:<16}{r['theta']:>11.4f}{t_static:>13.4f}"
                  f"{r['theta'] / t_static:>6.2f}x{r['gap']:>7.0e}{r['runtime']:>9.2f}")
            rows.append(dict(tm=label, theta_milp=r["theta"], theta_static=t_static,
                             mip_gap=r["gap"], runtime=r["runtime"],
                             T=T.round(6).tolist(), X=r["X"].tolist()))
            if rep == 0 and args.show_topologies:
                print(f"    x (links i->j):\n{fmt(r['X'])}")
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "part2_results%s.json" % ("_simple" if args.simple else ""))
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tm", help="CSV file with an 8x8 hose traffic matrix")
    ap.add_argument("--simple", action="store_true",
                    help="at most one link per ordered pair (no parallel links)")
    ap.add_argument("--robust", type=int, default=0, metavar="K",
                    help="one topology for K random hose TMs (max-min theta)")
    ap.add_argument("--reps", type=int, default=2, help="samples per TM family")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120)
    ap.add_argument("--show-topologies", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="Gurobi log")
    args = ap.parse_args()

    if args.tm:
        T = np.loadtxt(args.tm, delimiter=",")
        d = 4
        if not is_hose(T, d):
            print("WARNING: TM is not in the hose set (row/col sums <= 4, zero diag)")
        r = best_topology_milp(T, d, args.simple, args.time_limit, args.verbose)
        print(f"theta* = {r['theta']:.6f}   (MIP gap {r['gap']:.1e}, {r['runtime']:.2f}s)")
        print("x[i][j] = number of links i -> j:")
        print(fmt(r["X"]))
        print("link utilisation (flow / capacity):")
        util = np.divide(r["load"], r["X"], out=np.zeros_like(r["load"]), where=r["X"] > 0)
        print("\n".join("      " + " ".join(f"{v:4.2f}" for v in row) for row in util))
    elif args.robust:
        rng = np.random.default_rng(args.seed)
        Ts = [GENERATORS["mix_of_3_perms"](8, 4, rng) for _ in range(args.robust)]
        r = robust_topology_milp(Ts, 4, args.simple, args.time_limit, args.verbose)
        print(f"worst-case theta over {args.robust} TMs = {r['theta']:.4f}  (MIP gap {r['gap']:.1e}, {r['runtime']:.1f}s)")
        print("per-TM theta:", [round(t, 4) for t in r["thetas"]])
        for k, T in enumerate(Ts):
            own = best_topology_milp(T, 4, args.simple, args.time_limit)["theta"]
            print(f"  TM{k}: shared topology {r['thetas'][k]:.4f}  vs  its own best {own:.4f}")
        print("x:\n" + fmt(r["X"]))
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
