"""Basis-pursuit (L1-minimization) recovery via linear programming."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_basis_pursuit(A: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Solve  minimize ||x||_1  s.t.  A x = y  via the standard
    split-variable LP reformulation x = x_plus - x_minus, x_plus, x_minus >= 0:

        minimize  sum(x_plus) + sum(x_minus)
        s.t.      A x_plus - A x_minus = y
                  x_plus, x_minus >= 0

    Returns the recovered x in R^n, or None if the LP solver fails to
    converge (rare for well-posed random instances but can happen at
    numerically extreme aspect ratios).
    """
    m, n = A.shape
    if y.shape != (m,):
        raise ValueError(f"y must have shape ({m},), got {y.shape}")

    c = np.ones(2 * n)
    A_eq = np.hstack([A, -A])
    bounds = [(0, None)] * (2 * n)

    result = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if not result.success:
        return None
    return result.x[:n] - result.x[n:]


def check_recovery(x_hat: np.ndarray | None, x0: np.ndarray, tol: float = 1e-5) -> bool:
    """True if x_hat exists and matches x0 within tol in the sup norm."""
    if x_hat is None:
        return False
    return bool(np.max(np.abs(x_hat - x0)) < tol)
