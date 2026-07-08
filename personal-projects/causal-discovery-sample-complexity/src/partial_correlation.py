"""Partial correlation and the Fisher z conditional-independence test."""

import numpy as np
from scipy.stats import norm


def partial_correlation(cov: np.ndarray, i: int, j: int, cond_set) -> float:
    """Partial correlation of variables i, j given the variables in cond_set,
    computed from a covariance matrix by inverting the relevant submatrix.

    corr(i, j | S) = -Theta[0,1] / sqrt(Theta[0,0] * Theta[1,1])

    where Theta is the inverse of cov restricted to rows/cols {i, j} + S.
    """
    idx = [i, j] + list(cond_set)
    sub = cov[np.ix_(idx, idx)]
    try:
        precision = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(sub)
    denom = np.sqrt(precision[0, 0] * precision[1, 1])
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    r = -precision[0, 1] / denom
    return float(np.clip(r, -0.999999, 0.999999))


def fisher_z_test(r: float, n: int, cond_set_size: int, alpha: float) -> bool:
    """Fisher z test for conditional independence (partial correlation = 0).

    Returns True if the null hypothesis of independence is NOT rejected
    (i.e., the pair is judged conditionally independent given the data).
    """
    dof = n - cond_set_size - 3
    if dof <= 0:
        # Not enough samples to test this conditioning set: conservatively
        # assume dependence (do not remove the edge).
        return False
    z = 0.5 * np.log((1 + r) / (1 - r))
    stat = np.sqrt(dof) * abs(z)
    # Two-sided normal critical value.
    threshold = norm.ppf(1 - alpha / 2)
    return stat < threshold


def conditional_independence_test(
    cov: np.ndarray, i: int, j: int, cond_set, n: int, alpha: float
) -> bool:
    """True if i and j are judged independent given cond_set at level alpha."""
    r = partial_correlation(cov, i, j, cond_set)
    return fisher_z_test(r, n, len(cond_set), alpha)
