"""Percolation observables (giant component fraction, susceptibility).

These are the classical statistical-physics order parameters used to
locate a percolation phase transition: the fraction of nodes in the
largest connected component, and the mean finite-cluster size
("susceptibility"), which is expected to peak at the critical point in
the same way magnetic susceptibility peaks at a continuous phase
transition.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def component_sizes(self) -> np.ndarray:
        n = len(self.parent)
        roots = np.array([self.find(i) for i in range(n)])
        _, counts = np.unique(roots, return_counts=True)
        return counts


def percolation_curve(
    dist: np.ndarray, thresholds: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Giant-component fraction and susceptibility over a threshold grid.

    ``thresholds`` must be sorted ascending. Susceptibility is the mean
    squared cluster size *excluding* the largest cluster, normalized by
    n: chi(t) = sum_{clusters != giant} s^2 / n. It is the standard
    finite-size proxy for the percolation critical point.
    """
    n = dist.shape[0]
    iu = np.triu_indices(n, k=1)
    weights = dist[iu]
    order = np.argsort(weights)
    edges = list(zip(iu[0][order], iu[1][order]))
    edge_weights = weights[order]

    thresholds = np.asarray(thresholds, dtype=float)
    giant_frac = np.zeros(len(thresholds))
    susceptibility = np.zeros(len(thresholds))

    uf = UnionFind(n)
    ptr = 0
    num_edges = len(edges)
    for idx, t in enumerate(thresholds):
        while ptr < num_edges and edge_weights[ptr] <= t:
            a, b = edges[ptr]
            uf.union(a, b)
            ptr += 1
        sizes = uf.component_sizes()
        giant_idx = np.argmax(sizes)
        giant_frac[idx] = sizes[giant_idx] / n
        rest = np.delete(sizes, giant_idx)
        susceptibility[idx] = float((rest.astype(float) ** 2).sum() / n)

    return giant_frac, susceptibility


def find_percolation_threshold(
    dist: np.ndarray, thresholds: Sequence[float]
) -> float:
    """Return the threshold that maximizes susceptibility."""
    thresholds = np.asarray(thresholds, dtype=float)
    _, chi = percolation_curve(dist, thresholds)
    return float(thresholds[np.argmax(chi)])
