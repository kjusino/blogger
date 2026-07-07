"""Vietoris-Rips persistent homology of a coupled random-graph filtration."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from ripser import ripser


def compute_persistence(
    dist: np.ndarray, maxdim: int = 1, thresh: float = np.inf
) -> List[np.ndarray]:
    """Compute persistence diagrams for dimensions 0..maxdim.

    Returns a list of ``(k, 2)`` arrays of (birth, death) pairs, one per
    homology dimension.
    """
    result = ripser(dist, distance_matrix=True, maxdim=maxdim, thresh=thresh)
    return result["dgms"]


def betti_curve(dgm: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Number of bars alive (birth <= t < death) at each threshold t."""
    thresholds = np.asarray(thresholds, dtype=float)
    if dgm.size == 0:
        return np.zeros(len(thresholds))
    births = dgm[:, 0][:, None]
    deaths = dgm[:, 1][:, None]
    alive = (births <= thresholds[None, :]) & (deaths > thresholds[None, :])
    return alive.sum(axis=0).astype(float)
