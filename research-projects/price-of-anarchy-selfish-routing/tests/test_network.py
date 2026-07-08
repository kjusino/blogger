import numpy as np
import pytest

from src.cost_functions import Polynomial
from src.network import (
    Edge,
    Network,
    braess_network,
    pigou_network,
    random_series_parallel,
    three_parallel_links,
    two_link_pigou_style,
)


def test_incidence_matrix_simple_two_link():
    net = pigou_network()
    A = net.incidence_matrix()
    assert A.shape == (2, 2)
    # both edges go 0 -> 1: enter node 1 (+1), leave node 0 (-1)
    assert np.array_equal(A[:, 0], np.array([-1.0, 1.0]))
    assert np.array_equal(A[:, 1], np.array([-1.0, 1.0]))


def test_conservation_rhs():
    net = pigou_network()
    b = net.conservation_rhs()
    assert b[net.source] == pytest.approx(-1.0)
    assert b[net.sink] == pytest.approx(1.0)


def test_source_equals_sink_rejected():
    with pytest.raises(ValueError):
        Network(2, [Edge(0, 1, Polynomial((1.0,)))], source=0, sink=0)


def test_edge_endpoint_out_of_range_rejected():
    with pytest.raises(ValueError):
        Network(2, [Edge(0, 5, Polynomial((1.0,)))], source=0, sink=1)


def test_braess_network_shapes():
    net = braess_network(with_shortcut=False)
    assert net.n_nodes == 4 and net.n_edges == 4
    net2 = braess_network(with_shortcut=True)
    assert net2.n_nodes == 4 and net2.n_edges == 5


def test_three_parallel_links_shape():
    net = three_parallel_links()
    assert net.n_edges == 3
    assert all(e.u == net.source and e.v == net.sink for e in net.edges)


def test_two_link_pigou_style_edge_costs():
    net = two_link_pigou_style(degree=3, b=0.7)
    assert net.edges[0].cost(2.0) == pytest.approx(8.0)  # x^3 at x=2
    assert net.edges[1].cost(42.0) == pytest.approx(0.7)  # constant edge


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_random_series_parallel_is_connected_dag(degree):
    rng = np.random.default_rng(123)
    for _ in range(30):
        net = random_series_parallel(rng, degree=degree, max_edges=12)
        fwd, bwd = net.reachable_and_coreachable()
        assert net.sink in fwd
        assert net.source in bwd
        # every edge must lie on some source-sink path (no dangling edges)
        for e in net.edges:
            assert e.u in fwd and e.v in bwd
        # every edge should have the requested degree
        for e in net.edges:
            assert e.cost.degree == degree
        # DAG check: edges never point from a node back toward the source's
        # own predecessors in this construction; verify no self loops at least
        assert all(e.u != e.v for e in net.edges)


def test_random_series_parallel_respects_edge_budget():
    rng = np.random.default_rng(7)
    for _ in range(20):
        net = random_series_parallel(rng, degree=1, max_edges=10)
        assert 1 <= net.n_edges <= 10
