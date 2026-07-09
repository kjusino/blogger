"""Random-hyperplane (SimHash) locality-sensitive hashing, implemented from
scratch with only numpy (no LSH/ANN library)."""
from __future__ import annotations

import math

import numpy as np


def random_hyperplanes(k: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """k random hyperplane normals in R^dim, drawn i.i.d. from N(0, I).

    The normal need not be unit length: sign(r.x) is scale-invariant in r.
    """
    return rng.standard_normal(size=(k, dim))


def hash_bits(X: np.ndarray, hyperplanes: np.ndarray) -> np.ndarray:
    """Sign bits of X @ hyperplanes.T, as a boolean (n, k) array.

    X: (n, dim) array of vectors (rows need not be pre-normalized: only the
    sign of the projection matters).
    """
    projections = X @ hyperplanes.T
    return projections >= 0.0


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    """Pack a (n, k) boolean array into n Python ints (bucket keys)."""
    k = bits.shape[1]
    weights = 1 << np.arange(k)
    return (bits.astype(np.int64) @ weights).astype(np.int64)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Angle in [0, pi] between two vectors (not required to be unit norm)."""
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def empirical_single_bit_collision_rate(
    u: np.ndarray, v: np.ndarray, num_trials: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Monte Carlo estimate of Pr[sign(r.u) == sign(r.v)] over `num_trials`
    independent random hyperplanes.

    Returns (empirical_rate, standard_error) where standard_error is the
    binomial standard error sqrt(p*(1-p)/n) using the empirical p.
    """
    dim = u.shape[0]
    hyperplanes = random_hyperplanes(num_trials, dim, rng)
    bits_u = hash_bits(u[None, :], hyperplanes)[0]
    bits_v = hash_bits(v[None, :], hyperplanes)[0]
    agree = bits_u == bits_v
    p_hat = float(np.mean(agree))
    stderr = math.sqrt(p_hat * (1.0 - p_hat) / num_trials)
    return p_hat, stderr
