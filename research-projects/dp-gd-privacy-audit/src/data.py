"""Synthetic binary-classification dataset for the DP-GD privacy audit.

Two isotropic Gaussian blobs in d dimensions, n points total, deterministic
given an explicitly-seeded numpy Generator. No global numpy random state is
ever touched (no np.random.seed()) -- every RNG draw in this project is
threaded explicitly through the call chain from a caller-supplied
np.random.Generator.
"""
from __future__ import annotations

import numpy as np


def make_dataset(
    n: int = 100,
    d: int = 5,
    rng: np.random.Generator | None = None,
    separation: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate n points in d dims, two-Gaussian-blob binary classification.

    Class 0 is centered at -separation/2 * e0, class 1 at +separation/2 * e0
    (e0 = first standard basis vector), both with unit isotropic covariance.
    Classes are split as evenly as possible (n//2 and n - n//2).

    Args:
        n: number of points.
        d: number of feature dimensions.
        rng: an explicitly-seeded np.random.Generator (required; there is no
            default global-state fallback by design).
        separation: distance between the two blob centers along the first
            coordinate axis.

    Returns:
        X: (n, d) float64 feature matrix.
        y: (n,) float64 label vector in {0.0, 1.0}.
    """
    if rng is None:
        raise ValueError(
            "rng must be an explicitly-seeded np.random.Generator; "
            "this project never uses unseeded/global numpy randomness."
        )
    n0 = n // 2
    n1 = n - n0

    center0 = np.zeros(d)
    center1 = np.zeros(d)
    center0[0] = -separation / 2.0
    center1[0] = separation / 2.0

    X0 = rng.normal(loc=center0, scale=1.0, size=(n0, d))
    X1 = rng.normal(loc=center1, scale=1.0, size=(n1, d))

    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n0), np.ones(n1)])

    # Shuffle so class order isn't trivially grouped (deterministic given rng).
    perm = rng.permutation(n)
    return X[perm], y[perm]
