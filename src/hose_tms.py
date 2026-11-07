"""
Traffic-matrix generators for the hose set

    T_hose = { T >= 0 : T_ii = 0, sum_j T_ij <= d, sum_i T_ij <= d }   (d = 4)

Every generator returns an n x n numpy array with zero diagonal whose row and
column sums are exactly d (the hose set's "saturated" corners and interior).
"""
import numpy as np


def derangement(n, rng):
    while True:
        p = rng.permutation(n)
        if (p != np.arange(n)).all():
            return p


def perm_matrix(p):
    n = len(p)
    P = np.zeros((n, n))
    P[np.arange(n), p] = 1
    return P


def uniform(n, d, rng=None):
    T = np.full((n, n), d / (n - 1))
    np.fill_diagonal(T, 0)
    return T


def permutation(n, d, rng):
    """Every node sends all d units to ONE other node (hardest for multi-hop)."""
    return d * perm_matrix(derangement(n, rng))


def mix_of_permutations(n, d, rng, k=3):
    """Birkhoff-style random hose TM: random convex mix of k derangements."""
    w = rng.dirichlet(np.ones(k))
    return d * sum(wi * perm_matrix(derangement(n, rng)) for wi in w)


def random_hose(n, d, rng, iters=500):
    """Dense random TM scaled (Sinkhorn) to row/col sums d."""
    T = rng.exponential(size=(n, n)) ** 2
    np.fill_diagonal(T, 0)
    for _ in range(iters):
        T *= d / T.sum(axis=1, keepdims=True)
        T *= d / T.sum(axis=0, keepdims=True)
    return T


def clustered(n, d, rng, size=4):
    """Nodes form random groups of `size`; traffic stays inside a group
    (like a rack-local / all-reduce workload)."""
    perm = rng.permutation(n)
    T = np.zeros((n, n))
    for g in range(0, n, size):
        grp = perm[g:g + size]
        for i in grp:
            for j in grp:
                if i != j:
                    T[i, j] = d / (len(grp) - 1)
    return T


GENERATORS = {
    "uniform": uniform,
    "permutation": permutation,
    "mix_of_3_perms": mix_of_permutations,
    "random_dense": random_hose,
    "clustered_4": clustered,
    "clustered_2": lambda n, d, rng: clustered(n, d, rng, size=2),
}


def is_hose(T, d, tol=1e-4):
    return ((T >= -tol).all() and np.allclose(np.diag(T), 0)
            and (T.sum(axis=1) <= d + tol).all()
            and (T.sum(axis=0) <= d + tol).all())
