"""Network generators and spectral/degree statistics used to compute the
two mean-field epidemic-threshold predictions:

- QMF  (quenched mean-field / NIMFA, Wang et al. 2003):  tau_c = 1 / lambda_max(A)
- HMF  (heterogeneous / degree-based mean-field, Pastor-Satorras & Vespignani
  2001): tau_c = <k> / <k^2>

lambda_max(A) is computed with a from-scratch power iteration (rather than
calling straight into a library eigensolver) so it can be unit-tested against
hand-derived answers on small graphs, with a comparison against
numpy.linalg.eigvalsh as a correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp


def generate_random_regular(n: int, k: int, seed: int) -> nx.Graph:
    """k-regular random graph (homogeneous degree distribution)."""
    return nx.random_regular_graph(k, n, seed=seed)


def generate_erdos_renyi(n: int, mean_degree: float, seed: int) -> nx.Graph:
    """G(n, p) with p chosen to hit the target mean degree, restricted to
    its largest connected component (SIS thresholds are only meaningful on a
    single connected component)."""
    p = mean_degree / (n - 1)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return _largest_component(G)


def generate_barabasi_albert(n: int, mean_degree: float, seed: int) -> nx.Graph:
    """Scale-free network via preferential attachment. m is rounded from the
    target mean degree (BA mean degree -> 2m as n -> infinity)."""
    m = max(1, round(mean_degree / 2))
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return _largest_component(G)


def _largest_component(G: nx.Graph) -> nx.Graph:
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    H = G.subgraph(largest).copy()
    return nx.convert_node_labels_to_integers(H)


def power_iteration_lambda_max(A: sp.spmatrix, tol: float = 1e-10, max_iter: int = 10_000,
                                seed: int = 0) -> float:
    """Largest eigenvalue of a symmetric non-negative matrix A via power
    iteration. Perron-Frobenius guarantees lambda_max is real and, for a
    connected graph, simple -- but plain power iteration on A itself can
    fail to converge on a *bipartite* graph's adjacency matrix, since its
    spectrum is symmetric about 0 and -lambda_max then ties lambda_max in
    magnitude, leaving the iteration oscillating between the two
    eigenvectors. Diagonal-shifting by the Gershgorin row-sum bound before
    iterating (and un-shifting the result) makes every eigenvalue of the
    shifted matrix non-negative, so ranking by magnitude and by value agree
    and the dominant eigenvector is unambiguous regardless of bipartiteness."""
    n = A.shape[0]
    row_sums = np.asarray(np.abs(A).sum(axis=1)).ravel()
    shift = float(row_sums.max()) if n > 0 else 0.0
    A_shifted = A + shift * sp.identity(n, format="csr")

    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    lam_prev = 0.0
    for _ in range(max_iter):
        w = A_shifted @ v
        norm = np.linalg.norm(w)
        if norm == 0.0:
            return -shift
        v = w / norm
        lam = float(v @ (A_shifted @ v))
        if abs(lam - lam_prev) < tol * max(1.0, abs(lam)):
            return lam - shift
        lam_prev = lam
    return lam_prev - shift


@dataclass
class NetworkStats:
    n: int
    mean_degree: float
    mean_sq_degree: float
    heterogeneity_ratio: float  # <k^2> / <k>^2 ; = 1 for perfectly regular
    lambda_max: float
    qmf_threshold: float  # tau_c = 1 / lambda_max
    hmf_threshold: float  # tau_c = <k> / <k^2>


def compute_network_stats(G: nx.Graph, seed: int = 0) -> NetworkStats:
    n = G.number_of_nodes()
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    mean_k = degrees.mean()
    mean_k2 = (degrees ** 2).mean()
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    lambda_max = power_iteration_lambda_max(A, seed=seed)
    return NetworkStats(
        n=n,
        mean_degree=mean_k,
        mean_sq_degree=mean_k2,
        heterogeneity_ratio=mean_k2 / (mean_k ** 2),
        lambda_max=lambda_max,
        qmf_threshold=1.0 / lambda_max,
        hmf_threshold=mean_k / mean_k2,
    )


def build_network(topology: str, n: int, mean_degree: float, seed: int) -> nx.Graph:
    if topology == "rr":
        k = max(2, round(mean_degree))
        if (n * k) % 2 != 0:
            k += 1
        return generate_random_regular(n, k, seed)
    if topology == "er":
        return generate_erdos_renyi(n, mean_degree, seed)
    if topology == "ba":
        return generate_barabasi_albert(n, mean_degree, seed)
    raise ValueError(f"unknown topology: {topology}")
