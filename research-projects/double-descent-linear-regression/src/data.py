"""Synthetic isotropic linear-regression data generation.

Model: x ~ N(0, I_p), y = x . beta0 + eps, eps ~ N(0, sigma2), independent of x.
beta0 is fixed with ||beta0||^2 = r2 (signal energy held constant across p).
"""

import numpy as np


def generate_beta0(p: int, r2: float) -> np.ndarray:
    """Fixed coefficient vector with ||beta0||^2 = r2, embedded in R^p.

    Isotropy of the design makes the choice of direction irrelevant to any
    risk computed here, so the signal is placed on the first coordinate.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    beta0 = np.zeros(p)
    beta0[0] = np.sqrt(r2)
    return beta0


def sample_dataset(n: int, p: int, beta0: np.ndarray, sigma2: float,
                    rng: np.random.Generator):
    """Draw one training set (X, y) of n isotropic Gaussian examples in R^p."""
    if beta0.shape[0] != p:
        raise ValueError("beta0 dimension must match p")
    X = rng.standard_normal((n, p))
    eps = rng.standard_normal(n) * np.sqrt(sigma2)
    y = X @ beta0 + eps
    return X, y
