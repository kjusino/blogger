"""Correlated ±1 pattern generator for Hopfield-network experiments.

Generation model
-----------------
For a correlation parameter rho in [0, 1) and pattern length N, patterns are
generated as

    xi^mu_i = sign( sqrt(rho) * z_shared_i + sqrt(1 - rho) * z^mu_i )

where ``z_shared`` ~ N(0, 1)^N is drawn once per (N, rho) condition (a shared
"template" direction) and ``z^mu`` ~ N(0, 1)^N is drawn i.i.d. per pattern mu.
rho = 0 recovers i.i.d. random +-1 patterns (the classical Amit-Gutfreund-
Sompolinsky setting).

Arcsin law
----------
For jointly Gaussian X, Y with correlation rho, the sign nonlinearity gives

    corr(sign(X), sign(Y)) = (2 / pi) * arcsin(rho).

This module exposes ``arcsin_law`` so the relation can be checked numerically
against empirically generated patterns (see tests/test_patterns.py and
Figure (d) in the experiment report) rather than merely assumed.
"""

from __future__ import annotations

import numpy as np


def arcsin_law(rho: float) -> float:
    """Theoretical output +-1 correlation for input Gaussian correlation rho."""
    return (2.0 / np.pi) * np.arcsin(rho)


def generate_correlated_patterns(
    n: int,
    p: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw P correlated +-1 patterns of length N.

    Parameters
    ----------
    n : pattern length (number of neurons).
    p : number of patterns to draw.
    rho : correlation parameter in [0, 1). A single shared Gaussian template
        is drawn once and mixed into every pattern with weight sqrt(rho).
    rng : numpy Generator, so draws are reproducible given a fixed seed.

    Returns
    -------
    ndarray of shape (p, n), dtype float64, entries in {-1, +1}.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    if n <= 0 or p <= 0:
        raise ValueError("n and p must be positive integers")

    z_shared = rng.standard_normal(n)
    z_indep = rng.standard_normal((p, n))

    mixed = np.sqrt(rho) * z_shared[None, :] + np.sqrt(1.0 - rho) * z_indep
    # sign(0) is measure-zero for continuous Gaussians; map to +1 for safety.
    patterns = np.where(mixed >= 0.0, 1.0, -1.0)
    return patterns


def empirical_pairwise_correlation(patterns: np.ndarray) -> float:
    """Mean pairwise correlation (1/N) sum_i xi^mu_i xi^nu_i over all
    distinct pattern pairs (mu, nu), mu != nu. This is the quantity the
    arcsin law predicts as a function of the input Gaussian rho.
    """
    p, n = patterns.shape
    if p < 2:
        raise ValueError("need at least 2 patterns to compute pairwise correlation")
    # Gram matrix of normalized (per-N) overlaps.
    gram = (patterns @ patterns.T) / n
    off_diag_sum = gram.sum() - np.trace(gram)
    num_pairs = p * (p - 1)
    return off_diag_sum / num_pairs
