import numpy as np

from src.graph import UnionFind, is_connected, num_components, rips_edges


def test_rips_edges_finds_close_pairs():
    points = np.array([[0.1, 0.1], [0.15, 0.1], [0.9, 0.9]])
    edges = rips_edges(points, r=0.1)
    assert edges == [(0, 1)]


def test_rips_edges_wraps_around_periodic_boundary():
    # These two points are far apart in plain Euclidean distance but adjacent
    # once you wrap around the torus (0.01 and 0.99 are 0.02 apart on a torus).
    points = np.array([[0.01, 0.5], [0.99, 0.5]])
    edges = rips_edges(points, r=0.05)
    assert edges == [(0, 1)]
    edges_no_wrap_needed = rips_edges(points, r=0.9)
    assert edges_no_wrap_needed == [(0, 1)]  # sanity: still found without wraparound mattering


def test_rips_edges_empty_when_no_points():
    assert rips_edges(np.zeros((0, 2)), r=0.5) == []


def test_rips_edges_rejects_negative_radius():
    import pytest

    with pytest.raises(ValueError):
        rips_edges(np.zeros((2, 2)), r=-1.0)


def test_union_find_basic_merging():
    uf = UnionFind(5)
    assert uf.num_components == 5
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2)
    assert uf.num_components == 3
    assert uf.union(0, 2) is False  # already connected


def test_num_components_no_edges_all_isolated():
    assert num_components(4, []) == 4


def test_num_components_path_graph_is_one_component():
    assert num_components(4, [(0, 1), (1, 2), (2, 3)]) == 1


def test_num_components_two_disjoint_pairs():
    assert num_components(4, [(0, 1), (2, 3)]) == 2


def test_is_connected_single_vertex_trivially_connected():
    assert is_connected(1, []) is True


def test_is_connected_false_when_isolated_vertex_present():
    assert is_connected(3, [(0, 1)]) is False
