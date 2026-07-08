"""Random sensing-matrix and sparse-signal generation for compressed-sensing
recovery trials."""

from __future__ import annotations

import numpy as np


def gaussian_sensing_matrix(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """An m x n sensing matrix with iid N(0, 1/m) entries.

    The 1/m scaling normalizes columns to unit expected squared norm, the
    standard convention in the compressed-sensing phase-transition
    literature (keeps y = A x0 at an O(1) scale regardless of m).
    """
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive")
    return rng.standard_normal((m, n)) / np.sqrt(m)


def sparse_signal(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """A vector in R^n with exactly k nonzero entries at random positions,
    each drawn iid N(0, 1) (nonzero with probability 1)."""
    if k < 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 <= k <= n={n}")
    x = np.zeros(n)
    if k == 0:
        return x
    support = rng.choice(n, size=k, replace=False)
    x[support] = rng.standard_normal(k)
    return x
