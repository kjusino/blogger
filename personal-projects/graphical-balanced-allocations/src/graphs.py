"""Graph family generators for the graphical balanced-allocations experiment.

Each generator returns a connected, simple, undirected networkx.Graph on
exactly ``n`` nodes labeled ``0..n-1``, together with a compact
``(E, 2)`` numpy array of its edges (used by the allocation simulator so it
never has to touch the networkx object in the hot loop).
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh as sp_eigsh

FAMILIES = (
    "complete",
    "regular3",
    "regular10",
    "erdos_renyi",
    "smallworld_low_rewiring",
    "smallworld_high_rewiring",
    "torus",
    "cycle",
    "path",
)


def _edge_array(G):
    edges = np.array(list(G.edges()), dtype=np.int64)
    # canonicalize to make the array deterministic regardless of nx's
    # internal edge ordering / (u, v) vs (v, u) representation
    edges.sort(axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[order]


def generate_graph(family: str, n: int, seed: int):
    """Return (G, edges) for the named family on n nodes.

    Raises ValueError for an unknown family and RuntimeError if the
    resulting graph is not connected (the allocation process assumes a
    single connected component).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown graph family: {family!r}")

    rng = np.random.default_rng(seed)

    if family == "complete":
        G = nx.complete_graph(n)
    elif family == "regular3":
        d = 3 if n > 3 else n - 1
        if (n * d) % 2 == 1:
            d -= 1
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(2**31)))
    elif family == "regular10":
        d = 10 if n > 10 else n - 1
        if (n * d) % 2 == 1:
            d -= 1
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(2**31)))
    elif family == "erdos_renyi":
        p = min(1.0, 2.0 * np.log(n) / n)
        G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(2**31)))
    elif family == "smallworld_low_rewiring":
        k = 4 if n > 4 else n - 1
        G = nx.watts_strogatz_graph(n, k, 0.01, seed=int(rng.integers(2**31)))
    elif family == "smallworld_high_rewiring":
        k = 4 if n > 4 else n - 1
        G = nx.watts_strogatz_graph(n, k, 1.0, seed=int(rng.integers(2**31)))
    elif family == "torus":
        side = max(2, int(round(np.sqrt(n))))
        G = nx.grid_2d_graph(side, side, periodic=True)
        G = nx.convert_node_labels_to_integers(G)
        # grid_2d_graph(side, side) yields side*side nodes, which may not
        # equal the requested n exactly; that's fine, callers use G's
        # actual node count (see experiment.py) rather than assuming n.
    elif family == "cycle":
        G = nx.cycle_graph(n)
    elif family == "path":
        G = nx.path_graph(n)

    if G.number_of_nodes() < 2:
        raise RuntimeError(f"{family} graph on n={n} has fewer than 2 nodes")
    if not nx.is_connected(G):
        raise RuntimeError(f"{family} graph on n={n} (seed={seed}) is not connected")

    return G, _edge_array(G)


def spectral_gap(G) -> float:
    """Algebraic connectivity of the normalized Laplacian (in [0, 2]).

    This is the standard scalar measure of a graph's expansion: values
    bounded away from 0 indicate a good expander, values shrinking with n
    indicate poor expansion (e.g. paths/cycles have gap = Theta(1/n^2)).

    networkx's default Lanczos-based `algebraic_connectivity` can fail to
    converge (ArpackNoConvergence) on graphs whose normalized-Laplacian
    spectrum is nearly degenerate near 0 -- this happens in practice for
    paths and complete graphs, both used here. We use shift-invert ARPACK
    instead (fast and robust for the sparse random-graph families), with a
    dense LAPACK fallback (numerically exact, just slower) for whatever
    shift-invert still can't handle.
    """
    n = G.number_of_nodes()
    if n < 3:
        return float("nan")

    degrees = [d for _, d in G.degree()]
    n_edges = G.number_of_edges()
    if n_edges == n * (n - 1) // 2:
        # complete graph: exact closed form, and avoids a near-total-fill-in
        # sparse factorization that shift-invert would otherwise attempt
        return n / (n - 1)
    if n_edges == n and all(d == 2 for d in degrees):
        # cycle: normalized Laplacian eigenvalues are 1 - cos(2*pi*k/n)
        return 1.0 - np.cos(2.0 * np.pi / n)

    L = nx.normalized_laplacian_matrix(G).astype(np.float64)
    try:
        # shift-invert around a point just below the smallest possible
        # eigenvalue (0): turns "find the smallest eigenvalues" (slow,
        # ill-conditioned when they're clustered) into "find the largest
        # eigenvalues of the inverse" (fast, well-separated after the shift).
        eigvals = sp_eigsh(L, k=2, sigma=-1e-6, which="LM", return_eigenvectors=False)
        return float(np.sort(eigvals)[1])
    except Exception:
        dense = L.toarray()
        eigvals = np.linalg.eigvalsh(dense)
        return float(np.sort(eigvals)[1])
