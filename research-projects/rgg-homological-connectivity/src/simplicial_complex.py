"""Vietoris-Rips complex construction (vertices, edges, filled-in triangles).

The Rips complex at scale r contains a k-simplex for every (k+1)-subset of
points that are *pairwise* within distance r -- i.e. every simplex is a
clique of the r-neighborhood graph. We only need dimensions 0-2 to compute
Betti_0 and Betti_1, so "triangles" here means 3-cliques of the graph.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .gf2_linalg import row_from_indices
from .graph import rips_edges


@dataclass
class SimplicialComplex:
    n_vertices: int
    edges: List[Tuple[int, int]] = field(default_factory=list)
    triangles: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_triangles(self) -> int:
        return len(self.triangles)


def _enumerate_triangles(n: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Forward-neighbor triangle listing: O(sum_v deg_forward(v)^2).

    Standard trick (Schank & Wagner): only ever look "forward" (to
    higher-indexed neighbors), so each triangle is discovered exactly once,
    from its lowest-indexed vertex.
    """
    adj: List[set] = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    triangles = []
    for i in range(n):
        forward = sorted(x for x in adj[i] if x > i)
        for a in range(len(forward)):
            j = forward[a]
            adj_j = adj[j]
            for b in range(a + 1, len(forward)):
                k = forward[b]
                if k in adj_j:
                    triangles.append((i, j, k))
    return triangles


def build_complex(points: np.ndarray, r: float) -> SimplicialComplex:
    """Build the 2-skeleton of the Vietoris-Rips complex at radius r."""
    n = points.shape[0]
    edges = sorted(rips_edges(points, r))
    triangles = _enumerate_triangles(n, edges)
    return SimplicialComplex(n_vertices=n, edges=edges, triangles=triangles)


def boundary2_rows(complex_: SimplicialComplex) -> List[int]:
    """GF(2)-packed rows of the triangle -> edge boundary map (dim C_2 -> C_1).

    Row for triangle (i, j, k), i<j<k, has 1s in the columns of its three
    edges (i,j), (i,k), (j,k) -- the boundary of a 2-simplex over GF(2) is
    just the sum (= XOR) of its three faces.
    """
    edge_index: Dict[Tuple[int, int], int] = {e: idx for idx, e in enumerate(complex_.edges)}
    rows = []
    for i, j, k in complex_.triangles:
        rows.append(row_from_indices([edge_index[(i, j)], edge_index[(i, k)], edge_index[(j, k)]]))
    return rows
