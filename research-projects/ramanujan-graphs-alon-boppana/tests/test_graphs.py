import numpy as np
import pytest
import scipy.sparse as sp

from ramanujan_spectra.graphs import (
    complete_bipartite_regular_adjacency,
    complete_graph_adjacency,
    is_connected,
    networkx_regular_graph,
    pairing_model_regular_graph,
    petersen_graph_adjacency,
)


def _degree_sequence(adj: sp.spmatrix) -> np.ndarray:
    return np.asarray(adj.sum(axis=1)).ravel()


def _is_simple(adj: sp.spmatrix) -> bool:
    dense = adj.toarray()
    if np.any(np.diag(dense) != 0):
        return False
    if np.any((dense != 0) & (dense != 1)):
        return False
    return np.array_equal(dense, dense.T)


@pytest.mark.parametrize("d,n", [(3, 50), (4, 80), (3, 500), (4, 300)])
def test_pairing_model_produces_simple_d_regular_graph(d, n):
    rng = np.random.default_rng(0)
    adj = pairing_model_regular_graph(d, n, rng)
    assert adj.shape == (n, n)
    assert np.all(_degree_sequence(adj) == d)
    assert _is_simple(adj)


def test_pairing_model_is_deterministic_given_rng_state():
    adj_a = pairing_model_regular_graph(3, 60, np.random.default_rng(42))
    adj_b = pairing_model_regular_graph(3, 60, np.random.default_rng(42))
    assert (adj_a != adj_b).nnz == 0


def test_pairing_model_different_seeds_differ():
    adj_a = pairing_model_regular_graph(4, 100, np.random.default_rng(1))
    adj_b = pairing_model_regular_graph(4, 100, np.random.default_rng(2))
    assert (adj_a != adj_b).nnz > 0


def test_pairing_model_rejects_odd_handshake():
    with pytest.raises(ValueError):
        pairing_model_regular_graph(3, 5, np.random.default_rng(0))  # 3*5=15, odd


def test_pairing_model_rejects_n_leq_d():
    with pytest.raises(ValueError):
        pairing_model_regular_graph(5, 5, np.random.default_rng(0))


def test_pairing_model_gives_up_when_essentially_impossible():
    # d=10 has a vanishingly small chance of a random pairing being simple
    # (see graphs.py docstring); with very few restarts allowed this should
    # reliably fail rather than hang.
    with pytest.raises(RuntimeError):
        pairing_model_regular_graph(10, 200, np.random.default_rng(0), max_restarts=3)


@pytest.mark.parametrize("d,n", [(3, 50), (5, 120), (10, 300)])
def test_networkx_regular_graph_is_simple_d_regular(d, n):
    adj = networkx_regular_graph(d, n, seed=0)
    assert adj.shape == (n, n)
    assert np.all(_degree_sequence(adj) == d)
    assert _is_simple(adj)


def test_networkx_regular_graph_edge_count():
    d, n = 4, 100
    adj = networkx_regular_graph(d, n, seed=0)
    assert adj.sum() == d * n  # symmetric adjacency counts each edge twice


def test_is_connected_true_for_complete_graph():
    assert is_connected(complete_graph_adjacency(5))


def test_is_connected_false_for_disjoint_union():
    a = complete_graph_adjacency(3)
    block = sp.block_diag([a, a]).tocsr()
    assert not is_connected(block)


def test_complete_graph_adjacency_shape_and_degree():
    d = 6
    adj = complete_graph_adjacency(d)
    assert adj.shape == (d + 1, d + 1)
    assert np.all(_degree_sequence(adj) == d)


def test_complete_bipartite_regular_adjacency_shape_and_degree():
    d = 4
    adj = complete_bipartite_regular_adjacency(d)
    assert adj.shape == (2 * d, 2 * d)
    assert np.all(_degree_sequence(adj) == d)


def test_petersen_graph_adjacency_is_3_regular_on_10_vertices():
    adj = petersen_graph_adjacency()
    assert adj.shape == (10, 10)
    assert np.all(_degree_sequence(adj) == 3)
