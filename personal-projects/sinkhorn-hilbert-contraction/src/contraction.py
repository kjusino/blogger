"""Birkhoff/Hilbert-metric contraction rate for the Sinkhorn kernel K = exp(-C/eps).

Theory (Franklin & Lorenz, "On the scaling of multidimensional matrices",
Lin. Alg. Appl. 1989; see also Peyre & Cuturi, "Computational Optimal
Transport" (2019), Remark 4.22): for a positive matrix K, the map that
row/column-normalizes K to hit prescribed marginals is a contraction in
Hilbert's projective metric with Lipschitz constant

    eta(K) = tanh(Delta(K) / 4)

per single row- or column-projection, where Delta(K) is the projective
diameter

    Delta(K) = max_{i,j,k,l} log( K_ik K_jl / (K_il K_jk) ).

One full Sinkhorn "sweep" (a row projection followed by a column projection)
therefore contracts with rate kappa(K) = eta(K)^2 = tanh(Delta(K)/4)^2, since
Delta(K) = Delta(K^T) (the max is symmetric under swapping the roles of rows
and columns).

For K = exp(-C/eps), Delta(K) = D(C) / eps where

    D(C) = max_{i,j,k,l} (C_il + C_jk - C_ik - C_jl)

is a purely combinatorial functional of the cost matrix (independent of
eps). Both a brute-force O(n^2 m^2) evaluator and an O(n m^2) evaluator of
D(C) are provided below and cross-checked against each other in the test
suite; the fast one is what the experiment sweep actually uses.
"""
from __future__ import annotations

import numpy as np


def cost_diameter_brute_force(C: np.ndarray) -> float:
    """O(n^2 m^2) reference implementation of D(C) = max_{i,j,k,l}(C_il+C_jk-C_ik-C_jl)."""
    n, m = C.shape
    best = -np.inf
    for i in range(n):
        for j in range(n):
            # For fixed i,j, maximize over k,l independently:
            # (C_il - C_ik) + (C_jk - C_jl) = (C_il + C_jk) - (C_ik + C_jl)
            for k in range(m):
                for l in range(m):
                    val = C[i, l] + C[j, k] - C[i, k] - C[j, l]
                    if val > best:
                        best = val
    return float(best)


def cost_diameter_fast(C: np.ndarray) -> float:
    """O(n m^2) vectorized evaluator of D(C), exploiting separability in i and j.

    D(C) = max_{k,l} [ a(k,l) + a(l,k) ],  a(k,l) := max_i ( C_il - C_ik ).
    """
    n, m = C.shape
    # diff[i, k, l] = C[i, l] - C[i, k]
    diff = C[:, None, :] - C[:, :, None]  # shape (n, m, m)
    a = diff.max(axis=0)  # shape (m, m); a[k, l] = max_i (C_il - C_ik)
    return float((a + a.T).max())


def hilbert_diameter(K: np.ndarray) -> float:
    """Delta(K) computed directly from a positive matrix K (used for the
    Sinkhorn-kernel invariance test: Delta(exp(-C/eps)) should equal
    cost_diameter_fast(C)/eps for every eps).
    """
    log_k = np.log(K)
    n, m = K.shape
    diff = log_k[:, None, :] - log_k[:, :, None]
    a = diff.max(axis=0)
    return float((a + a.T).max())


def theoretical_contraction_rate(C: np.ndarray, eps: float) -> float:
    """kappa_theory(K) = tanh(Delta(K)/4)^2 for K = exp(-C/eps), i.e. the
    Birkhoff upper bound on the per-sweep Hilbert-metric contraction ratio.
    """
    delta = cost_diameter_fast(C) / eps
    eta = np.tanh(delta / 4.0)
    return float(eta * eta)
