"""Small statistics helpers: bootstrap confidence intervals and transition-width
extraction from a connectivity-fraction curve.
"""

from typing import Callable, Tuple

import numpy as np


def bootstrap_ci(
    samples,
    rng: np.random.Generator,
    statistic: Callable = np.mean,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) via the nonparametric bootstrap."""
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    if n == 0:
        raise ValueError("need at least one sample")
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = statistic(samples[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(statistic(samples)), float(lo), float(hi)


def transition_width(
    r_grid: np.ndarray, frac_connected: np.ndarray, low: float = 0.1, high: float = 0.9
) -> Tuple[float, float, float]:
    """Radius interval over which the connected-fraction curve rises from `low` to `high`.

    Percolation-style transition sharpness measure: assumes frac_connected is
    (weakly) monotone increasing in r, which holds here because adding edges
    (increasing r) can never disconnect a graph.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    frac = np.asarray(frac_connected, dtype=float)
    r_low = float(np.interp(low, frac, r_grid))
    r_high = float(np.interp(high, frac, r_grid))
    return r_high - r_low, r_low, r_high
