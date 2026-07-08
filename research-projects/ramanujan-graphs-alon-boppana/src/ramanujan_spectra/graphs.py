"""Random d-regular graph generation.

Two independent generators are provided so the spectral results below don't
rest on trusting a single implementation:

1. ``pairing_model_regular_graph`` -- a from-scratch implementation of the
   classical configuration ("pairing") model with full restart on failure
   (Bollobas 1980): pair up d*n stubs uniformly at random, and if the result
   has a self-loop or a repeated edge, throw the whole pairing away and try
   again. Conditional on producing a simple graph, this is *exactly* uniform
   over labeled simple d-regular graphs on n vertices -- there is no
   approximation here, only rejection. For fixed d, as n -> infinity the
   probability that a single random pairing is already simple converges to
   a positive constant depending only on d (Bollobas 1980), but that
   constant shrinks very quickly as d grows (empirically: ~14% for d=3,
   ~2% for d=4, well under 0.5% by d=6, and unusably small by d=10 -- see
   ``results/generator_cross_validation.json``). This generator is used in
   this project only for small/moderate d, as an independent cross-check of
   the library generator below (``experiment.compare_generators``), not for
   the main sweep.

2. ``networkx_regular_graph`` -- networkx's ``random_regular_graph``, a
   trusted, independently-written library implementation that additionally
   repairs a bad pairing via targeted edge swaps rather than discarding it
   outright, so it stays fast for every d used in this project (including
   d=10, where full-restart rejection is hopeless). This is the generator
   used for the main sweep (``experiment.run_sweep``).

Both return a symmetric 0/1 SciPy sparse adjacency matrix.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp


def _edges_to_sparse_adjacency(lo: np.ndarray, hi: np.ndarray, n: int) -> sp.csr_matrix:
    rows = np.concatenate([lo, hi])
    cols = np.concatenate([hi, lo])
    data = np.ones(rows.shape[0], dtype=np.float64)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    return adj


def pairing_model_regular_graph(
    d: int, n: int, rng: np.random.Generator, max_restarts: int = 2000
) -> sp.csr_matrix:
    """Sample a simple d-regular graph on n vertices via the pairing model.

    Raises ValueError for degenerate/impossible parameters and RuntimeError
    if no simple graph was found within max_restarts attempts (should never
    happen in practice for the (d, n) pairs used in this project).
    """
    if d < 1:
        raise ValueError("d must be >= 1")
    if n <= d:
        raise ValueError("n must be > d for a simple d-regular graph to exist")
    if (d * n) % 2 != 0:
        raise ValueError("d * n must be even (handshake lemma)")

    for _ in range(max_restarts):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        u = stubs[0::2]
        v = stubs[1::2]

        if np.any(u == v):
            continue

        lo = np.minimum(u, v)
        hi = np.maximum(u, v)
        keys = lo.astype(np.int64) * n + hi.astype(np.int64)
        if np.unique(keys).shape[0] != keys.shape[0]:
            continue

        return _edges_to_sparse_adjacency(lo, hi, n)

    raise RuntimeError(
        f"pairing model failed to produce a simple graph in {max_restarts} "
        f"restarts (d={d}, n={n})"
    )


def networkx_regular_graph(d: int, n: int, seed: int) -> sp.csr_matrix:
    """Sample a d-regular graph on n vertices using networkx's generator."""
    g = nx.random_regular_graph(d, n, seed=seed)
    return nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)


def is_connected(adj: sp.csr_matrix) -> bool:
    n_components = sp.csgraph.connected_components(adj, directed=False, return_labels=False)
    return n_components == 1


# --- known small graphs with exact, hand-verifiable spectra, used as unit
# test fixtures for the eigenvalue code (not part of the random-graph study).

def complete_graph_adjacency(d: int) -> sp.csr_matrix:
    """K_{d+1}: d-regular; spectrum {d (x1), -1 (x d)}."""
    g = nx.complete_graph(d + 1)
    return nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)


def complete_bipartite_regular_adjacency(d: int) -> sp.csr_matrix:
    """K_{d,d}: d-regular; spectrum {d (x1), -d (x1), 0 (x 2d-2)}."""
    g = nx.complete_bipartite_graph(d, d)
    return nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)


def petersen_graph_adjacency() -> sp.csr_matrix:
    """The Petersen graph: 3-regular on 10 vertices, strongly regular
    (10, 3, 0, 1); spectrum {3 (x1), 1 (x5), -2 (x4)}. A famous example of
    an (optimal, Moore-graph-adjacent) Ramanujan graph: lambda(G) = 2 <=
    2*sqrt(2) = 2.828..."""
    g = nx.petersen_graph()
    return nx.to_scipy_sparse_array(g, format="csr", dtype=np.float64)
