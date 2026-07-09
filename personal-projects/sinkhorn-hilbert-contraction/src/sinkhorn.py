"""Log-domain (numerically stabilized) Sinkhorn algorithm for entropic optimal
transport, with per-iteration marginal-residual tracking.

Naive Sinkhorn (u,v <- r./Kv, c./K'u with K = exp(-C/eps)) underflows to all
zeros in K for small eps, which is exactly the regime this experiment cares
about (small eps => large condition number => the interesting contraction
rates). So iteration is done entirely in the dual-potential / log domain,
following Peyre & Cuturi, "Computational Optimal Transport" (2019) Remark 4.7.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class SinkhornResult:
    plan: np.ndarray          # (n, m) transport plan P
    f: np.ndarray              # row dual potential
    g: np.ndarray               # column dual potential
    residual_history: np.ndarray  # max(row_l1_err, col_l1_err) after each full sweep
    n_iter: int
    converged: bool


def sinkhorn_log(
    C: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    eps: float,
    max_iter: int = 10_000,
    tol: float = 1e-10,
) -> SinkhornResult:
    """Run stabilized Sinkhorn to (near-)convergence.

    A "full sweep" = one f-update followed by one g-update. The residual
    recorded per sweep is the L1 marginal violation of the plan implied by
    the *current* (f, g), which is the quantity Sinkhorn theory (Franklin &
    Lorenz 1989) predicts decays geometrically once transients die out.
    """
    n, m = C.shape
    if r.shape != (n,) or c.shape != (m,):
        raise ValueError("marginals must match cost matrix shape")
    if not np.isclose(r.sum(), 1.0) or not np.isclose(c.sum(), 1.0):
        raise ValueError("marginals must be probability vectors (sum to 1)")
    if eps <= 0:
        raise ValueError("eps must be positive")

    log_r = np.log(r)
    log_c = np.log(c)
    f = np.zeros(n)
    g = np.zeros(m)

    history = np.empty(max_iter)
    n_iter = max_iter
    converged = False
    log_plan = None

    for it in range(max_iter):
        f = eps * (log_r - logsumexp((g[None, :] - C) / eps, axis=1))
        g = eps * (log_c - logsumexp((f[:, None] - C) / eps, axis=0))

        log_plan = (f[:, None] + g[None, :] - C) / eps
        row_sums = np.exp(logsumexp(log_plan, axis=1))
        col_sums = np.exp(logsumexp(log_plan, axis=0))
        err = max(np.abs(row_sums - r).sum(), np.abs(col_sums - c).sum())
        history[it] = err

        if err < tol:
            n_iter = it + 1
            converged = True
            break

    plan = np.exp(log_plan)
    return SinkhornResult(
        plan=plan,
        f=f,
        g=g,
        residual_history=history[:n_iter],
        n_iter=n_iter,
        converged=converged,
    )


def entropic_cost(plan: np.ndarray, C: np.ndarray) -> float:
    """<P, C> -- the (non-regularized) transport cost of a plan."""
    return float(np.sum(plan * C))
