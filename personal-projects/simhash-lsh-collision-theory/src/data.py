"""Synthetic vector generators with exactly-controlled pairwise angles.

Using synthetic, angle-controlled data (rather than a real embedding
dataset) is deliberate: it lets every experiment compare against the exact
theoretical angle theta, with no confound from an unknown/estimated ground
truth similarity distribution.
"""
from __future__ import annotations

import numpy as np


def random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def random_unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    V = rng.standard_normal((n, dim))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return V / norms


def vector_at_angle(u: np.ndarray, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Construct a unit vector v with angle EXACTLY `theta` from unit vector u.

    Method: draw a random vector, Gram-Schmidt it to be orthogonal to u and
    normalize (call it w), then set v = cos(theta) * u + sin(theta) * w.
    Since u.w = 0 and both are unit norm, u.v = cos(theta) exactly (up to
    floating point), which is verified in tests/test_data.py.
    """
    dim = u.shape[0]
    raw = rng.standard_normal(dim)
    raw = raw - np.dot(raw, u) * u
    norm = np.linalg.norm(raw)
    if norm < 1e-10:
        # Degenerate draw (raw ~ parallel to u); resample once, deterministically
        # perturbed, rather than looping (astronomically unlikely in dim > 1).
        raw = rng.standard_normal(dim) + 0.1
        raw = raw - np.dot(raw, u) * u
        norm = np.linalg.norm(raw)
    w = raw / norm
    v = np.cos(theta) * u + np.sin(theta) * w
    return v / np.linalg.norm(v)


def planted_neighbor_dataset(
    n_background: int, dim: int, near_theta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """A query vector, a background of `n_background` uniformly random unit
    vectors (i.i.d., i.e. "far" points w.h.p. in high dimension), plus one
    planted near neighbor at angle exactly `near_theta` from the query,
    inserted at a random position.

    Returns (query, dataset, planted_index).
    """
    query = random_unit_vector(dim, rng)
    background = random_unit_vectors(n_background, dim, rng)
    planted = vector_at_angle(query, near_theta, rng)
    planted_index = int(rng.integers(0, n_background + 1))
    dataset = np.insert(background, planted_index, planted, axis=0)
    return query, dataset, planted_index
