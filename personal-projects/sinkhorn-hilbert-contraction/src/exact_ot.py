"""Exact (non-regularized) optimal transport cost for balanced, uniform-marginal
problems, via the Hungarian algorithm -- used as ground truth to sanity-check
that Sinkhorn's entropic cost converges to the true OT cost as eps -> 0.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def exact_ot_uniform(C: np.ndarray) -> float:
    """Exact OT cost for square C with uniform marginals r = c = ones(n)/n.

    For a balanced transportation problem with equal uniform marginals on
    both sides, the transportation polytope's extreme points are exactly
    (1/n) * permutation matrices, so the LP optimum coincides with (1/n)
    times the optimal assignment cost -- solved exactly by the Hungarian
    algorithm (scipy's linear_sum_assignment).
    """
    n, m = C.shape
    if n != m:
        raise ValueError("exact_ot_uniform requires a square cost matrix")
    row_ind, col_ind = linear_sum_assignment(C)
    return float(C[row_ind, col_ind].sum()) / n
