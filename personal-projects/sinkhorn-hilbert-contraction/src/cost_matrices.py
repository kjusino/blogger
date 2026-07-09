"""Cost-matrix generators for the Sinkhorn contraction-rate experiment.

Each generator returns a plain (n, m) numpy array of nonnegative transport
costs C, plus a short metadata string identifying the family. Marginals are
always taken to be uniform (r = ones(n)/n, c = ones(m)/m) elsewhere -- the
generators here only build C.
"""
from __future__ import annotations

import numpy as np


def random_points_cost(n: int, m: int, rng: np.random.Generator, dim: int = 2) -> np.ndarray:
    """Squared-Euclidean cost between two independent uniform point clouds in [0,1]^dim."""
    x = rng.uniform(0.0, 1.0, size=(n, dim))
    y = rng.uniform(0.0, 1.0, size=(m, dim))
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=-1)


def clustered_points_cost(
    n: int,
    m: int,
    rng: np.random.Generator,
    dim: int = 2,
    n_clusters: int = 2,
    cluster_std: float = 0.05,
    separation: float = 0.6,
) -> np.ndarray:
    """Squared-Euclidean cost between two point clouds, each drawn from
    `n_clusters` well-separated Gaussian blobs placed on a regular simplex-ish
    layout of radius `separation`. Structured / low-effective-rank cost, in
    contrast to `random_points_cost`.
    """
    centers = np.zeros((n_clusters, dim))
    for k in range(n_clusters):
        angle = 2.0 * np.pi * k / n_clusters
        centers[k, 0] = separation * np.cos(angle)
        if dim > 1:
            centers[k, 1] = separation * np.sin(angle)

    def sample(count: int) -> np.ndarray:
        assign = rng.integers(0, n_clusters, size=count)
        pts = centers[assign] + rng.normal(0.0, cluster_std, size=(count, dim))
        return pts

    x = sample(n)
    y = sample(m)
    diff = x[:, None, :] - y[None, :, :]
    return np.sum(diff * diff, axis=-1)


def grid_1d_cost(n: int, m: int) -> np.ndarray:
    """|i/n - j/m| cost on two evenly spaced 1-D grids on [0,1] (deterministic, no rng)."""
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, m)
    return np.abs(x[:, None] - y[None, :])


def iid_random_cost(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. Uniform(0,1) cost entries -- no Euclidean/metric structure at all.

    Serves as a structural negative control: the Birkhoff bound only looks at
    the matrix K = exp(-C/eps), so this family tests whether "genericness" of
    C (as opposed to any geometric meaning) is what governs bound tightness.
    """
    return rng.uniform(0.0, 1.0, size=(n, m))


FAMILIES = {
    "random_points": random_points_cost,
    "clustered_points": clustered_points_cost,
    "grid_1d": grid_1d_cost,
    "iid_random": iid_random_cost,
}


def build_cost(family: str, n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    if family == "random_points":
        return random_points_cost(n, m, rng)
    if family == "clustered_points":
        return clustered_points_cost(n, m, rng)
    if family == "grid_1d":
        return grid_1d_cost(n, m)
    if family == "iid_random":
        return iid_random_cost(n, m, rng)
    raise ValueError(f"unknown cost family: {family!r}")
