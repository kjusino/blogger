"""Extremal-eigenvalue computation for d-regular graph adjacency matrices.

The quantity of interest throughout this project is the standard
"second eigenvalue" of a d-regular graph in the Ramanujan-graph sense:

    lambda(G) = max_{i >= 2} |lambda_i(G)|

i.e. the largest absolute value among all eigenvalues *other* than the top
one (lambda_1 = d, achieved by the all-ones vector on a connected regular
graph). This needs both the second-largest algebraic eigenvalue and the
most negative one, since a "bad" graph can fail to be Ramanujan from either
end of the spectrum -- most dramatically, a bipartite d-regular graph
always has lambda_min = -d exactly, so lambda(G) = d (the worst possible
value), regardless of anything else about its second-largest eigenvalue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

DENSE_CUTOFF = 300


@dataclass(frozen=True)
class SpectralSummary:
    n: int
    d: int
    lambda1: float
    lambda2: float
    lambda_min: float
    lambda2_abs: float  # max(lambda2, |lambda_min|)
    bipartite_like: bool


def _dense_extremes(adj: sp.spmatrix | np.ndarray) -> np.ndarray:
    dense = adj.toarray() if sp.issparse(adj) else np.asarray(adj)
    return np.linalg.eigvalsh(dense)  # ascending


def _sparse_extremes(adj: sp.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    k_eff = min(k, n - 2)
    ncv = min(n - 1, max(4 * k_eff + 1, 40))
    adj = adj.astype(np.float64)
    try:
        top = eigsh(adj, k=k_eff, which="LA", ncv=ncv, return_eigenvectors=False)
        bottom = eigsh(adj, k=k_eff, which="SA", ncv=ncv, return_eigenvectors=False)
    except ArpackNoConvergence:
        ncv2 = min(n - 1, max(8 * k_eff + 1, 80))
        top = eigsh(adj, k=k_eff, which="LA", ncv=ncv2, maxiter=5000, return_eigenvectors=False)
        bottom = eigsh(adj, k=k_eff, which="SA", ncv=ncv2, maxiter=5000, return_eigenvectors=False)
    return np.sort(top), np.sort(bottom)


def extremal_eigenvalues(
    adj: sp.spmatrix, d: int, k: int = 4, bipartite_tol: float = 1e-6
) -> SpectralSummary:
    """Compute lambda1, lambda2, lambda_min (and the derived lambda2_abs)
    for a d-regular graph's adjacency matrix."""
    n = adj.shape[0]

    if n <= DENSE_CUTOFF:
        eigvals = _dense_extremes(adj)  # ascending, full spectrum
        lambda1 = float(eigvals[-1])
        lambda2 = float(eigvals[-2])
        lambda_min = float(eigvals[0])
    else:
        top_sorted, bottom_sorted = _sparse_extremes(adj, k=k)
        lambda1 = float(top_sorted[-1])
        lambda2 = float(top_sorted[-2])
        lambda_min = float(bottom_sorted[0])

    if abs(lambda1 - d) > 1e-4 * max(1, d):
        raise ValueError(
            f"top eigenvalue {lambda1} is not close to d={d}; graph may not "
            "be d-regular or the eigensolver failed to converge correctly"
        )

    lambda2_abs = max(lambda2, abs(lambda_min))
    bipartite_like = abs(lambda_min + d) < bipartite_tol

    return SpectralSummary(
        n=n,
        d=d,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_min=lambda_min,
        lambda2_abs=lambda2_abs,
        bipartite_like=bipartite_like,
    )
