"""Random point clouds on the flat unit torus.

We sample on a torus (periodic unit square) rather than the plain unit
square so that every point has a statistically identical neighborhood --
no boundary effects to correct for. This matches the setting used in the
random-geometric-graph / random-geometric-complex literature (Penrose 2003;
Kahle 2011) when they state clean asymptotic threshold formulas.
"""

import numpy as np


def sample_torus_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n points i.i.d. uniform on the unit torus [0, 1)^2.

    Returns an (n, 2) float64 array.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return rng.random((n, 2))


def torus_pairwise_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimum-image displacement vectors between two point sets on the unit torus.

    a: (m, 2), b: (k, 2) -> (m, k, 2) displacement using periodic wraparound.
    """
    diff = a[:, None, :] - b[None, :, :]
    return diff - np.round(diff)
