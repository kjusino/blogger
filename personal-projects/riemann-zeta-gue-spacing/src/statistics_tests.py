"""Statistical comparisons between an empirical unfolded zero sequence and
the reference distributions in gue_theory.py."""

import numpy as np
from scipy.stats import kstest


def ks_against_cdf(samples, cdf_fn):
    """One-sample Kolmogorov-Smirnov test of `samples` against a
    theoretical CDF callable. Returns (statistic, p_value); a large
    p_value means the data is consistent with that CDF."""
    samples = np.asarray(samples, dtype=float)
    result = kstest(samples, cdf_fn)
    return float(result.statistic), float(result.pvalue)


def pair_correlation_histogram(unfolded_positions, u_max=3.0, n_bins=20):
    """Empirical pair-correlation function R2_hat(u) for an unfolded
    (unit-mean-spacing) point sequence.

    Edge-corrected: a start point x_i only contributes if the window
    [x_i, x_i + u_max] lies entirely inside the observed range, so every
    bin is averaged over the same fully-eligible set of start points
    rather than silently undercounting near the right edge of the window.

    Returns (bin_centers, r2_hat). For a Poisson process r2_hat -> 1
    everywhere; for a repulsive process (e.g. GUE) it is suppressed near
    u=0.
    """
    x = np.asarray(unfolded_positions, dtype=float)
    if u_max <= 0 or n_bins < 1:
        raise ValueError("u_max must be positive and n_bins >= 1")
    edges = np.linspace(0.0, u_max, n_bins + 1)
    du = edges[1] - edges[0]
    counts = np.zeros(n_bins)

    eligible = x[x <= x[-1] - u_max]
    n_eligible = len(eligible)
    if n_eligible == 0:
        raise ValueError(
            f"window spans only {x[-1] - x[0]:.2f} unfolded units, "
            f"too small for u_max={u_max}"
        )

    for xi in eligible:
        diffs = x[(x > xi) & (x <= xi + u_max)] - xi
        idx = np.minimum((diffs / du).astype(int), n_bins - 1)
        np.add.at(counts, idx, 1)

    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    r2_hat = counts / (n_eligible * du)
    return bin_centers, r2_hat


def pair_correlation_l2_error(bin_centers, r2_hat, theory_fn):
    """Root-mean-square distance between an empirical pair-correlation
    histogram and a theoretical R2(u) callable, evaluated at the same
    bin centers."""
    theory = theory_fn(bin_centers)
    return float(np.sqrt(np.mean((r2_hat - theory) ** 2)))


def repulsion_fraction(spacings_arr, threshold=0.2):
    """Fraction of spacings below `threshold`. Level repulsion drives this
    toward 0 for GUE (spacing density ~ s^2 near s=0), while an
    uncorrelated Poisson process keeps it near 1 - exp(-threshold)."""
    s = np.asarray(spacings_arr, dtype=float)
    return float(np.mean(s < threshold))
