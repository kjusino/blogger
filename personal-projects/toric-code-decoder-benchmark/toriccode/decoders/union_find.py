"""A simplified, Union-Find-inspired clustering decoder.

This is *not* the optimal O(n alpha(n)) peeling decoder of Delfosse &
Nickerson (Quantum 5, 595, 2021). It captures the same core idea -- grow
clusters of nearby defects with a union-find structure until every cluster
has even defect parity, then only pay the (expensive) exact-matching cost
*within* each small cluster instead of over the full defect set -- using a
simpler, radius-thresholded clustering rule that is easier to verify for
correctness. See the project README for the honest scope of this
simplification and how it differs from the literature algorithm.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..stabilizer import ToricLattice
from .mwpm import match_defects


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_defects(defects: list[tuple[int, int]], L: int, max_radius: int | None = None):
    """Grow a union-find clustering of defects: repeatedly increase a shared
    radius threshold and merge any pair of defects (in different clusters)
    whose torus distance is within that radius, stopping as soon as every
    cluster has an even number of defects.

    Returns a list of clusters, each a list of indices into `defects`.
    """
    m = len(defects)
    if m == 0:
        return []
    if max_radius is None:
        max_radius = 2 * L

    dist = [
        [ToricLattice.torus_distance(defects[i], defects[j], L) for j in range(m)]
        for i in range(m)
    ]
    uf = _UnionFind(m)

    def cluster_counts():
        counts = defaultdict(int)
        for i in range(m):
            counts[uf.find(i)] += 1
        return counts

    radius = 1
    while radius <= max_radius:
        counts = cluster_counts()
        if all(v % 2 == 0 for v in counts.values()):
            break
        for i in range(m):
            for j in range(i + 1, m):
                if dist[i][j] <= radius and uf.find(i) != uf.find(j):
                    uf.union(i, j)
        radius += 1

    groups = defaultdict(list)
    for i in range(m):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def decode(lattice: ToricLattice, syndrome: np.ndarray):
    """Return a correction (h_corr, v_corr) for the given syndrome using
    cluster-restricted matching."""
    L = lattice.L
    defects = lattice.defect_list(syndrome)
    h_corr = np.zeros((L, L), dtype=np.uint8)
    v_corr = np.zeros((L, L), dtype=np.uint8)
    if not defects:
        return h_corr, v_corr

    clusters = cluster_defects(defects, L)
    for group in clusters:
        if len(group) < 2:
            continue
        local_defects = [defects[i] for i in group]
        pairs = match_defects(local_defects, L)
        for i, j in pairs:
            toggles = lattice.dual_path_toggles(local_defects[i], local_defects[j])
            lattice.apply_toggles(h_corr, v_corr, toggles)
    return h_corr, v_corr
