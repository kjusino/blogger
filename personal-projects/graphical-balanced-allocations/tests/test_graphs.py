import networkx as nx
import numpy as np
import pytest

from src import graphs


@pytest.mark.parametrize("family", graphs.FAMILIES)
def test_generate_graph_is_connected_simple(family):
    G, edges = graphs.generate_graph(family, 64, seed=1)
    assert nx.is_connected(G)
    assert not any(u == v for u, v in edges)
    assert G.number_of_nodes() >= 2


@pytest.mark.parametrize("family", graphs.FAMILIES)
def test_generate_graph_deterministic_given_seed(family):
    G1, edges1 = graphs.generate_graph(family, 50, seed=7)
    G2, edges2 = graphs.generate_graph(family, 50, seed=7)
    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()
    np.testing.assert_array_equal(edges1, edges2)


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        graphs.generate_graph("not_a_family", 10, seed=0)


def test_edge_array_matches_networkx_edge_count():
    G, edges = graphs.generate_graph("cycle", 20, seed=0)
    assert edges.shape == (G.number_of_edges(), 2)
    assert edges.shape[0] == 20  # cycle has exactly n edges


def test_complete_graph_edge_count():
    n = 15
    G, edges = graphs.generate_graph("complete", n, seed=0)
    assert edges.shape[0] == n * (n - 1) // 2


def test_spectral_gap_ordering_matches_expansion_theory():
    # complete graph should have a much larger spectral gap than a cycle
    # of the same size -- this is the qualitative claim the whole
    # experiment rests on, so it gets a direct, cheap sanity check here.
    n = 200
    G_complete, _ = graphs.generate_graph("complete", n, seed=0)
    G_cycle, _ = graphs.generate_graph("cycle", n, seed=0)
    gap_complete = graphs.spectral_gap(G_complete)
    gap_cycle = graphs.spectral_gap(G_cycle)
    assert gap_complete > gap_cycle
    assert gap_cycle < 0.01  # cycle gap shrinks like Theta(1/n^2)


def test_spectral_gap_small_graph_returns_nan():
    G = nx.path_graph(2)
    assert np.isnan(graphs.spectral_gap(G))
