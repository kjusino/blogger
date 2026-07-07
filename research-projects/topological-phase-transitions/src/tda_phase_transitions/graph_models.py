"""Coupled random-graph generators.

Each generator returns a symmetric ``(n, n)`` "distance" matrix ``D`` (zero
diagonal) together with any auxiliary data needed for theoretical threshold
formulas. The matrices are constructed so that thresholding at a scalar
filtration parameter ``t`` (i.e. keeping edges with ``D[i, j] <= t``)
reproduces the corresponding classical random-graph model at parameter
``t``. Feeding the *full* matrix straight into a Vietoris-Rips persistent
homology routine (see ``persistence.py``) therefore computes the topology
of the entire growing-graph process in a single pass, rather than one
graph snapshot at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _symmetric_zero_diag(upper: np.ndarray, n: int) -> np.ndarray:
    """Build a symmetric matrix with zero diagonal from strict-upper values."""
    mat = np.zeros((n, n), dtype=upper.dtype)
    iu = np.triu_indices(n, k=1)
    mat[iu] = upper
    mat[(iu[1], iu[0])] = upper
    return mat


@dataclass
class ERModel:
    n: int
    dist: np.ndarray


def er_distance_matrix(n: int, seed: Optional[int] = None) -> ERModel:
    """Erdos-Renyi G(n, p), coupled over all p via i.i.d. Uniform(0, 1) labels.

    Edge (i, j) is present at threshold p iff dist[i, j] <= p, which is
    exactly the standard coupling used to construct G(n, p) for all p
    simultaneously on a shared probability space.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = _rng(seed)
    upper = rng.random(n * (n - 1) // 2)
    dist = _symmetric_zero_diag(upper, n)
    return ERModel(n=n, dist=dist)


@dataclass
class RGGModel:
    n: int
    dist: np.ndarray
    points: np.ndarray


def rgg_distance_matrix(n: int, seed: Optional[int] = None) -> RGGModel:
    """Random geometric graph on the unit torus [0, 1)^2.

    Edge (i, j) present at radius r iff the toroidal Euclidean distance
    between points i and j is <= r. This is precisely the RGG(n, r) model.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = _rng(seed)
    points = rng.random((n, 2))
    diff = np.abs(points[:, None, :] - points[None, :, :])
    diff = np.minimum(diff, 1.0 - diff)
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, 0.0)
    return RGGModel(n=n, dist=dist, points=points)


@dataclass
class ChungLuModel:
    n: int
    dist: np.ndarray
    weights: np.ndarray
    gamma: float


def _bounded_power_law_weights(
    n: int, gamma: float, w_min: float, w_max: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample n i.i.d. weights from a bounded Pareto-type power law.

    Density f(w) ~ w^(-gamma) on [w_min, w_max], gamma > 1, via inverse
    transform sampling of the truncated Pareto CDF.
    """
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for a normalizable power law")
    u = rng.random(n)
    a = w_min ** (1.0 - gamma)
    b = w_max ** (1.0 - gamma)
    return (a - u * (a - b)) ** (1.0 / (1.0 - gamma))


def chung_lu_distance_matrix(
    n: int,
    gamma: float = 2.5,
    w_min: float = 1.0,
    w_max_ratio: float = 10.0,
    seed: Optional[int] = None,
) -> ChungLuModel:
    """Chung-Lu (soft configuration) model with power-law weights.

    Coupled over the mean-degree scale ``theta`` via i.i.d. Uniform(0, 1)
    labels: edge (i, j) is present at threshold ``theta`` iff
    ``dist[i, j] <= theta``, which reproduces edge probability
    ``min(1, theta * w_i * w_j / L)`` with ``L = sum(weights)`` -- the
    defining property of the Chung-Lu random graph model.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = _rng(seed)
    w_max = w_min * w_max_ratio
    weights = _bounded_power_law_weights(n, gamma, w_min, w_max, rng)
    total = weights.sum()
    wi = weights[:, None]
    wj = weights[None, :]
    prod = (wi * wj) / total
    upper_u = rng.random(n * (n - 1) // 2)
    iu = np.triu_indices(n, k=1)
    prod_upper = prod[iu]
    # dist_ij = u_ij / prod_ij  =>  edge present at theta iff u_ij <= theta * prod_ij
    dist_upper = upper_u / prod_upper
    dist = _symmetric_zero_diag(dist_upper, n)
    return ChungLuModel(n=n, dist=dist, weights=weights, gamma=gamma)
