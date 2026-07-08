"""Generic statistics helpers: susceptibility-peak localization via
parabolic interpolation, block-bootstrap confidence intervals for
autocorrelated QS time series, relative-error metrics, and an exact
permutation-based Spearman correlation test (appropriate here since the
topology x size grid used for the heterogeneity-vs-error-gap test has few
enough points, n <= 12, that scipy's asymptotic t-approximation p-value is
unreliable, particularly near |rho_s| = 1)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.stats import spearmanr


def parabolic_peak(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Locate the peak of a unimodal-ish curve sampled at xs by parabolic
    (3-point) interpolation around the grid argmax. Falls back to the raw
    grid argmax if the argmax is at an endpoint (no interpolation possible)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    i = int(np.nanargmax(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), float(ys[i])

    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]

    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return float(x1), float(y1)

    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom

    if a >= 0:
        # Not a concave peak (noisy/monotonic segment) -- no interpolation.
        return float(x1), float(y1)

    x_peak = -b / (2 * a)
    if not (x0 <= x_peak <= x2):
        return float(x1), float(y1)

    c = y1 - a * x1 ** 2 - b * x1
    y_peak = a * x_peak ** 2 + b * x_peak + c
    return float(x_peak), float(y_peak)


def block_bootstrap_ci(series: np.ndarray, block_size: int, n_resamples: int = 2000,
                        ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of an autocorrelated series
    (the QS prevalence time series is a Markov chain, so naive i.i.d.
    resampling would understate the variance of the mean estimate)."""
    rng = np.random.default_rng(seed)
    n = len(series)
    n_blocks = max(1, n // block_size)
    starts_max = n - block_size
    if starts_max <= 0:
        return float(series.mean()), float(series.mean())

    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([series[s:s + block_size] for s in starts])
        boot_means[b] = sample.mean()

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(lo), float(hi)


def relative_error(theoretical: float, empirical: float) -> float:
    """|theoretical - empirical| / empirical. NaN if the empirical value is
    zero/undefined (no meaningful relative error to report)."""
    if empirical == 0 or np.isnan(empirical):
        return float("nan")
    return abs(theoretical - empirical) / empirical


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman rank correlation with an exact permutation p-value
    (brute-force over all n! orderings), valid for the small sample sizes
    used in this project. For n > 9 this falls back to scipy's asymptotic
    p-value (n! becomes too large to enumerate quickly)."""
    n = len(x)
    rho, asymptotic_p = spearmanr(x, y)
    if n > 9:
        return float(rho), float(asymptotic_p)

    x_ranks = np.argsort(np.argsort(x)).astype(float)
    y_ranks = np.argsort(np.argsort(y)).astype(float)

    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denom = np.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())
    observed = abs((x_centered * y_centered).sum() / denom)

    perm_idx = np.array(list(permutations(range(n))))  # (n!, n)
    y_perm_centered = y_centered[perm_idx]  # (n!, n)
    correlations = (y_perm_centered @ x_centered) / denom
    p_exact = float((np.abs(correlations) >= observed - 1e-9).mean())

    return float(rho), p_exact
