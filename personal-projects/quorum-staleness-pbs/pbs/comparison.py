"""Metrics for comparing two staleness curves P_stale(delta)."""

from __future__ import annotations

import numpy as np


def sup_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """Sup-norm distance between two staleness curves.

    Since P_stale(delta) = 1 - F(delta) for the (unobserved) random variable
    "time until the read quorum first overlaps the committed write", this is
    exactly a two-sample Kolmogorov-Smirnov statistic between the theoretical
    and empirical distributions of that random variable.
    """
    return float(np.max(np.abs(curve_a - curve_b)))


def area_between(curve_a: np.ndarray, curve_b: np.ndarray, deltas: np.ndarray) -> float:
    """L1 area between two staleness curves, integrated over delta via the trapezoid rule."""
    return float(np.trapezoid(np.abs(curve_a - curve_b), deltas))


def monte_carlo_noise_floor(n_trials: int, z: float = 3.0) -> float:
    """A conservative bound on Monte Carlo sampling noise for a single P_stale(delta) point.

    Worst-case binomial standard error is 0.5 / sqrt(n_trials) (at p=0.5);
    z=3 gives a ~3-sigma bound so genuine model divergence can be distinguished
    from sampling noise.
    """
    return z * 0.5 / np.sqrt(n_trials)
