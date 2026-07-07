"""Closed-form spectral theory for the coordinate-refresh walk on {0,1}^n.

The chain: at each step pick a coordinate i in [n] uniformly at random and
replace x_i with a fresh fair coin flip. Diagonalizing in the Fourier
(Walsh-Hadamard) basis of {0,1}^n, the characters chi_S(x) = prod_{i in S}
(-1)^{x_i} are exact eigenfunctions with eigenvalue lambda_{|S|} = 1 - |S|/n,
multiplicity C(n, |S|). This is the classical hypercube walk analyzed by
Diaconis, Graham & Morrison (1990) and Diaconis (1988, Ch. 6), which exhibits
a cutoff at t* = (n ln n)/2 with window Theta(n).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def eigenvalues(n: int) -> np.ndarray:
    """Return lambda_k = 1 - k/n for k = 0, ..., n."""
    if n < 1:
        raise ValueError("n must be a positive integer")
    k = np.arange(n + 1)
    return 1.0 - k / n


def chi_square(n: int, t) -> np.ndarray:
    """Exact chi-square distance squared: sum_{k=1}^n C(n,k) * (1-k/n)^(2t).

    t may be a scalar or array of nonnegative integers/floats. Computed in
    log-domain (logsumexp over log-binomial + 2t*log|lambda_k|) so it stays
    accurate for n in the thousands, where C(n, n/2) alone can exceed 1e300.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_arr < 0):
        raise ValueError("t must be nonnegative")

    k = np.arange(1, n + 1)  # drop k=0 (trivial eigenfunction, contributes 0)
    lam = 1.0 - k / n
    log_binom = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    with np.errstate(divide="ignore"):
        log_abs_lam = np.log(np.abs(lam))  # -inf exactly at k=n (lam=0)

    out = np.empty_like(t_arr)
    for i, tt in enumerate(t_arr):
        if tt == 0:
            # lambda_k^0 == 1 for every k, including k=n (0**0 by convention)
            log_lam_term = np.zeros_like(log_abs_lam)
        else:
            log_lam_term = 2 * tt * log_abs_lam
        log_terms = log_binom + log_lam_term
        out[i] = np.exp(logsumexp(log_terms))

    result = out if out.shape != (1,) or np.ndim(t) != 0 else out[0]
    return result


def tv_upper_bound(n: int, t) -> np.ndarray:
    """Diaconis-Shahshahani bound: TV(t) <= sqrt(chi_square(n, t)) / 2."""
    return 0.5 * np.sqrt(chi_square(n, t))


def cutoff_time(n: int) -> float:
    """Asymptotic cutoff location t*(n) = (n ln n) / 2."""
    if n < 2:
        raise ValueError("cutoff_time requires n >= 2")
    return 0.5 * n * np.log(n)


def rescale(n: int, t) -> np.ndarray:
    """Map real time t to the cutoff-window variable c = (2t - n ln n) / n.

    Chosen so that t = (n/2)(ln n + c); the cutoff prediction is that the
    mixing profile, plotted against c, converges to a single n-independent
    curve (`limiting_profile`) as n -> infinity.
    """
    if n < 2:
        raise ValueError("rescale requires n >= 2")
    t_arr = np.asarray(t, dtype=float)
    return (2 * t_arr - n * np.log(n)) / n


def unrescale(n: int, c) -> np.ndarray:
    """Inverse of `rescale`: map window variable c back to real time t."""
    if n < 2:
        raise ValueError("unrescale requires n >= 2")
    c_arr = np.asarray(c, dtype=float)
    return 0.5 * n * (np.log(n) + c_arr)


def limiting_profile(c) -> np.ndarray:
    """Universal chi-square-bound TV profile in the cutoff window: n -> infinity limit
    of tv_upper_bound(n, unrescale(n, c)), obtained from the small-k approximation
    C(n,k) ~ n^k/k! of the chi-square sum:

        chi_square(t) ~ exp(n * exp(-2t/n)) - 1 = exp(exp(-c)) - 1.
    """
    c_arr = np.asarray(c, dtype=float)
    return 0.5 * np.sqrt(np.expm1(np.exp(-c_arr)))
