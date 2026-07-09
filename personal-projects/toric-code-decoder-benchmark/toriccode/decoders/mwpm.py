"""Exact minimum-weight perfect matching (MWPM) decoder.

Builds the complete graph over syndrome defects with edge weight equal to the
torus (wrap-around Manhattan) distance, finds a minimum-weight perfect
matching via networkx's general-graph (blossom) algorithm, and reconstructs
the correction chain by walking a shortest dual-lattice path between each
matched pair.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..stabilizer import ToricLattice


def match_defects(defects: list[tuple[int, int]], L: int) -> list[tuple[int, int]]:
    """Return a list of (i, j) index pairs into `defects` forming a
    minimum-weight perfect matching under torus distance. Empty input yields
    an empty matching."""
    m = len(defects)
    if m == 0:
        return []
    if m % 2 != 0:
        raise ValueError("defect count must be even")

    graph = nx.Graph()
    graph.add_nodes_from(range(m))
    for i in range(m):
        for j in range(i + 1, m):
            dist = ToricLattice.torus_distance(defects[i], defects[j], L)
            graph.add_edge(i, j, weight=-dist)

    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    return [tuple(sorted(pair)) for pair in matching]


def decode(lattice: ToricLattice, syndrome: np.ndarray):
    """Return a correction (h_corr, v_corr) for the given syndrome."""
    L = lattice.L
    defects = lattice.defect_list(syndrome)
    h_corr = np.zeros((L, L), dtype=np.uint8)
    v_corr = np.zeros((L, L), dtype=np.uint8)
    if not defects:
        return h_corr, v_corr

    pairs = match_defects(defects, L)
    for i, j in pairs:
        toggles = lattice.dual_path_toggles(defects[i], defects[j])
        lattice.apply_toggles(h_corr, v_corr, toggles)
    return h_corr, v_corr
