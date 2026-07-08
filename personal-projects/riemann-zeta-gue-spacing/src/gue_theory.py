"""Reference distributions this experiment tests zeta zero statistics
against: the GUE Wigner-surmise spacing law, the Poisson (uncorrelated)
null, and Montgomery's conjectured pair-correlation form.
"""

import numpy as np
from scipy.special import erf


def gue_surmise_pdf(s):
    """GUE (beta=2) Wigner-surmise nearest-neighbor spacing density,
    normalized to unit mean spacing:

        p(s) = (32/pi^2) s^2 exp(-4 s^2 / pi)

    This finite-random-matrix approximation to the true (Fredholm
    determinant) GUE spacing law is accurate to within a fraction of a
    percent and is the standard comparison used in the numerical
    random-matrix-theory literature (Mehta, *Random Matrices*, 3rd ed.,
    ch. 6).
    """
    s = np.asarray(s, dtype=float)
    a = 4.0 / np.pi
    return (32.0 / np.pi**2) * s**2 * np.exp(-a * s**2)


def gue_surmise_cdf(s):
    """Closed-form antiderivative of gue_surmise_pdf:

        CDF(s) = erf(2s/sqrt(pi)) - (4s/pi) exp(-4 s^2/pi)

    Derived symbolically and checked against the pdf by differentiation
    (see tests/test_gue_theory.py).
    """
    s = np.asarray(s, dtype=float)
    a = 4.0 / np.pi
    return erf(2.0 * s / np.sqrt(np.pi)) - (4.0 * s / np.pi) * np.exp(-a * s**2)


def poisson_pdf(s):
    """Nearest-neighbor spacing density of an uncorrelated (Poisson)
    point process of unit density -- the null hypothesis GUE repulsion
    is tested against."""
    s = np.asarray(s, dtype=float)
    return np.exp(-s)


def poisson_cdf(s):
    s = np.asarray(s, dtype=float)
    return 1.0 - np.exp(-s)


def montgomery_pair_correlation(u):
    """Montgomery's 1973 conjectured pair-correlation form for the zeros
    of zeta (proved by Montgomery only for restricted Fourier support;
    the general case remains open):

        R2(u) = 1 - (sin(pi u) / (pi u))^2

    identical to the GUE sine-kernel two-level correlation function. The
    removable singularity at u=0 is taken as its limit, R2(0) = 0.
    """
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    nonzero = u != 0
    x = np.pi * u[nonzero]
    out[nonzero] = 1.0 - (np.sin(x) / x) ** 2
    return out
