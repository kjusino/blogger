"""Sampling from the rank-one spiked covariance model.

Sigma = I_p + lam * v v^T,  v a unit vector in R^p, lam >= 0 the spike
strength. Samples are drawn as x_i = Sigma^{1/2} z_i with z_i ~ N(0, I_p)
i.i.d., using the closed-form rank-one square root

    Sigma^{1/2} = I_p + (sqrt(1 + lam) - 1) * v v^T

which avoids ever forming or decomposing the p x p matrix Sigma.
"""

from __future__ import annotations

import numpy as np


def make_spike_direction(p: int, rng: np.random.Generator) -> np.ndarray:
    """A uniformly random unit vector in R^p."""
    v = rng.standard_normal(p)
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise ValueError("degenerate zero vector drawn; retry with a fresh rng")
    return v / norm


def sample_spiked_covariance_data(
    n: int,
    p: int,
    lam: float,
    rng: np.random.Generator,
    v: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n i.i.d. samples from N(0, I_p + lam * v v^T).

    Returns (X, v) where X has shape (n, p) and v is the unit spike
    direction used (drawn fresh if not provided).
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if v is None:
        v = make_spike_direction(p, rng)
    elif v.shape != (p,) or not np.isclose(np.linalg.norm(v), 1.0):
        raise ValueError("v must be a unit vector of shape (p,)")

    z = rng.standard_normal((n, p))
    alpha = np.sqrt(1.0 + lam) - 1.0
    proj = z @ v
    X = z + alpha * np.outer(proj, v)
    return X, v


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """The (biased, mean-known-zero) sample covariance (1/n) X^T X."""
    n = X.shape[0]
    return (X.T @ X) / n


def top_eigenpair(S: np.ndarray) -> tuple[float, np.ndarray]:
    """Largest eigenvalue and corresponding unit eigenvector of symmetric S."""
    eigvals, eigvecs = np.linalg.eigh(S)
    return float(eigvals[-1]), eigvecs[:, -1]
