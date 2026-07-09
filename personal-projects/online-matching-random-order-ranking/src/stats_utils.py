"""Small bootstrap-based statistics helpers -- no scipy.stats dependency,
just plain resampling, so the confidence intervals in this project make no
distributional assumptions on the (bounded, non-Gaussian) ratio samples."""

import numpy as np


def mean_ci(samples, confidence=0.95, n_bootstrap=2000, rng=None):
    """Bootstrap percentile confidence interval for the mean of `samples`.
    Returns (mean, lo, hi)."""
    samples = np.asarray(samples, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = len(samples)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(samples[0]), float(samples[0]), float(samples[0])
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = samples[idx].mean(axis=1)
    lo = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    hi = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return float(samples.mean()), float(lo), float(hi)


def two_sample_diff_ci(a, b, confidence=0.95, n_bootstrap=2000, rng=None):
    """Bootstrap CI for mean(a) - mean(b) with a, b independent samples
    (not assumed paired or equal length). If the interval excludes 0, the
    difference is significant at the given confidence level."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    na, nb = len(a), len(b)
    idx_a = rng.integers(0, na, size=(n_bootstrap, na))
    idx_b = rng.integers(0, nb, size=(n_bootstrap, nb))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    lo = np.percentile(diffs, (1 - confidence) / 2 * 100)
    hi = np.percentile(diffs, (1 + confidence) / 2 * 100)
    return float(a.mean() - b.mean()), float(lo), float(hi)


def linear_fit(x, y):
    """Ordinary least squares y ~ a*x + b. Returns (slope, intercept, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)
