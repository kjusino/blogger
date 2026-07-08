import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from src.networks import (
    build_network,
    compute_network_stats,
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_random_regular,
    power_iteration_lambda_max,
)


def test_power_iteration_matches_numpy_eig_on_random_symmetric_matrix():
    rng = np.random.default_rng(0)
    M = rng.random((40, 40))
    M = (M + M.T) / 2  # symmetric
    M = np.abs(M)  # non-negative, satisfies Perron-Frobenius preconditions
    expected = np.linalg.eigvalsh(M).max()
    got = power_iteration_lambda_max(sp.csr_matrix(M), seed=1)
    assert got == pytest.approx(expected, rel=1e-6)


def test_power_iteration_on_cycle_graph_matches_known_value():
    # lambda_max of a cycle graph C_n adjacency matrix is exactly 2.
    G = nx.cycle_graph(30)
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=float)
    got = power_iteration_lambda_max(A, seed=2)
    assert got == pytest.approx(2.0, abs=1e-6)


def test_complete_graph_qmf_and_hmf_coincide():
    # K_n is perfectly regular: lambda_max = n-1 = mean_degree, so
    # QMF = 1/(n-1) and HMF = (n-1)/(n-1)^2 = 1/(n-1) -- identical.
    G = nx.complete_graph(5)
    stats = compute_network_stats(G, seed=0)
    assert stats.lambda_max == pytest.approx(4.0, abs=1e-6)
    assert stats.mean_degree == pytest.approx(4.0)
    assert stats.mean_sq_degree == pytest.approx(16.0)
    assert stats.heterogeneity_ratio == pytest.approx(1.0)
    assert stats.qmf_threshold == pytest.approx(0.25, abs=1e-6)
    assert stats.hmf_threshold == pytest.approx(0.25, abs=1e-6)


def test_random_regular_generator_is_exactly_k_regular():
    G = generate_random_regular(100, 6, seed=3)
    degrees = [d for _, d in G.degree()]
    assert set(degrees) == {6}


def test_erdos_renyi_generator_returns_connected_graph():
    G = generate_erdos_renyi(300, mean_degree=6.0, seed=4)
    assert nx.is_connected(G)
    assert G.number_of_nodes() <= 300


def test_barabasi_albert_generator_returns_connected_graph():
    G = generate_barabasi_albert(300, mean_degree=6.0, seed=5)
    assert nx.is_connected(G)


def test_barabasi_albert_is_more_heterogeneous_than_random_regular():
    rr = generate_random_regular(500, 6, seed=6)
    ba = generate_barabasi_albert(500, mean_degree=6.0, seed=6)
    rr_stats = compute_network_stats(rr, seed=0)
    ba_stats = compute_network_stats(ba, seed=0)
    assert rr_stats.heterogeneity_ratio == pytest.approx(1.0, abs=1e-9)
    assert ba_stats.heterogeneity_ratio > rr_stats.heterogeneity_ratio


def test_build_network_dispatch_rejects_unknown_topology():
    with pytest.raises(ValueError):
        build_network("nonexistent", 10, 4.0, seed=0)


def test_build_network_matches_target_mean_degree_reasonably():
    for topo in ("rr", "er", "ba"):
        G = build_network(topo, 400, mean_degree=6.0, seed=7)
        degrees = np.array([d for _, d in G.degree()])
        assert 4.0 < degrees.mean() < 9.0
