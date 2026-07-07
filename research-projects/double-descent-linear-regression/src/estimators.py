"""Estimators for isotropic linear regression.

`fit_min_norm` covers both classical regimes with a single call: numpy's
`lstsq` returns the unique least-squares solution when n > p (OLS) and the
minimum-Euclidean-norm interpolating solution when n < p, via the SVD
pseudoinverse in both cases.
"""

import numpy as np


def fit_min_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta_hat, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    return beta_hat


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge estimator argmin ||y - X beta||^2 + lam ||beta||^2.

    lam == 0 falls back to `fit_min_norm` so the same code path also
    reproduces the ridgeless interpolator exactly at the boundary.
    """
    if lam == 0:
        return fit_min_norm(X, y)
    n, p = X.shape
    if p <= n:
        A = X.T @ X + lam * np.eye(p)
        b = X.T @ y
        return np.linalg.solve(A, b)
    # Woodbury form is cheaper and better conditioned when p > n.
    A = X @ X.T + lam * np.eye(n)
    return X.T @ np.linalg.solve(A, y)
