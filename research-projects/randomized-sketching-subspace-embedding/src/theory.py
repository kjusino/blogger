"""Theoretical predictions for random-sketch subspace embeddings, and small
statistics helpers (power-law fitting) used to test them against simulation.
"""

import numpy as np


def predicted_k(d, eps, delta=0.1, const=4.0):
    """Sample complexity sufficient for an eps-subspace embedding w.p. >= 1 - delta:
    k = C * (d + log(1/delta)) / eps^2, per the standard JL / subspace-embedding
    sample-complexity bound (e.g. Woodruff, "Sketching as a Tool for Numerical
    Linear Algebra", Thm 2.3). `const` is a loose, unoptimized constant -- the
    experiments treat the *scaling* (linear in d, quadratic in 1/eps) as the
    falsifiable claim, not this specific constant.
    """
    return int(np.ceil(const * (d + np.log(1.0 / delta)) / eps ** 2))


def subspace_distortion(SQ):
    """Exact worst-case distortion eps such that
        (1 - eps) ||x||^2 <= ||SQ x||^2 <= (1 + eps) ||x||^2   for all x,
    given Q has orthonormal columns (so ||Q x|| = ||x||). This is exactly
    max(1 - sigma_min(SQ)^2, sigma_max(SQ)^2 - 1) -- no sampling over x needed.
    """
    sv = np.linalg.svd(SQ, compute_uv=False)
    return max(1.0 - float(np.min(sv)) ** 2, float(np.max(sv)) ** 2 - 1.0)


def fit_power_law(x, y):
    """Least-squares fit of y = a * x^b in log-log space. Returns (a, b, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    b, log_a = np.polyfit(lx, ly, 1)
    a = np.exp(log_a)
    pred = log_a + b * lx
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(a), float(b), float(r2)
