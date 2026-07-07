"""Rips graph construction on the torus and union-find connectivity."""

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def rips_edges(points: np.ndarray, r: float) -> List[Tuple[int, int]]:
    """All pairs (i, j), i < j, within periodic distance r of each other.

    Uses scipy's periodic-box cKDTree (boxsize=1.0 on both axes) so this
    is O(n log n)-ish rather than the naive O(n^2) all-pairs scan.
    """
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}")
    if points.shape[0] == 0:
        return []
    tree = cKDTree(points, boxsize=1.0)
    pairs = tree.query_pairs(r, output_type="ndarray")
    return [(int(i), int(j)) for i, j in pairs]


class UnionFind:
    """Standard union-find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.num_components = n

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
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.num_components -= 1
        return True


def num_components(n: int, edges: List[Tuple[int, int]]) -> int:
    """Number of connected components of the graph (n vertices, given edges)."""
    uf = UnionFind(n)
    for i, j in edges:
        uf.union(i, j)
    return uf.num_components


def is_connected(n: int, edges: List[Tuple[int, int]]) -> bool:
    return n <= 1 or num_components(n, edges) == 1
