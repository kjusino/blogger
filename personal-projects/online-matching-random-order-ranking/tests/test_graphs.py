import numpy as np

from src.graphs import (complete_bipartite, perfect_matching_graph,
                         perturb_keep_perfect_matching, random_bipartite,
                         random_edge_flip, staircase_graph)
from src.matching import (matching_size, max_matching_size,
                           ranking_online_matching)


def test_complete_bipartite_has_all_edges():
    graph = complete_bipartite(4)
    assert graph.num_edges() == 16
    assert all((i, j) in graph.edges for i in range(4) for j in range(4))


def test_random_bipartite_edge_density_roughly_matches_p():
    rng = np.random.default_rng(0)
    n, p = 40, 0.3
    graph = random_bipartite(n, p, rng)
    density = graph.num_edges() / (n * n)
    assert abs(density - p) < 0.05  # generous tolerance, n*n=1600 samples


def test_random_bipartite_extremes():
    rng = np.random.default_rng(0)
    assert random_bipartite(5, 0.0, rng).num_edges() == 0
    assert random_bipartite(5, 1.0, rng).num_edges() == 25


def test_staircase_graph_has_perfect_matching_and_expected_degrees():
    n = 10
    graph = staircase_graph(n)
    assert max_matching_size(graph) == n
    # right-vertex j has degree j+1 (adjacent to left 0..j)
    for j in range(n):
        assert len(graph.adj_right[j]) == j + 1
    # left-vertex 0 is adjacent to every right vertex; left-vertex n-1 only to j=n-1
    assert len(graph.adj_left[0]) == n
    assert graph.adj_left[n - 1] == [n - 1]


def test_staircase_graph_with_identity_order_is_forced_regardless_of_rank():
    # Right vertex j is adjacent to left 0..j and arrives in index order, so
    # by induction each arrival has exactly one remaining unmatched option
    # (0..j-1 already forced to left 0..j-1) -- RANKING gets a perfect
    # matching with probability 1, no matter the random rank. This is why
    # `staircase_hard_instance` (src/experiment.py) pairs this graph with
    # the *reversed* arrival order instead, where that inductive argument
    # breaks: the widest-neighborhood vertex arrives first and can take any
    # offline vertex, including ones later (narrower) arrivals need.
    rng = np.random.default_rng(0)
    n = 12
    graph = staircase_graph(n)
    order = list(range(n))
    for _ in range(15):
        rank = rng.permutation(n)
        match = ranking_online_matching(graph, rank, order)
        assert matching_size(match) == n


def test_perfect_matching_graph_is_exactly_a_permutation():
    rng = np.random.default_rng(0)
    n = 9
    graph = perfect_matching_graph(n, rng)
    assert graph.num_edges() == n
    assert max_matching_size(graph) == n
    left_degrees = [len(row) for row in graph.adj_left]
    right_degrees = [len(row) for row in graph.adj_right]
    assert all(d == 1 for d in left_degrees)
    assert all(d == 1 for d in right_degrees)


def test_random_edge_flip_changes_exactly_one_edge():
    rng = np.random.default_rng(0)
    graph = staircase_graph(5)
    for _ in range(20):
        flipped = random_edge_flip(graph, rng)
        assert len(graph.edges.symmetric_difference(flipped.edges)) == 1


def test_perturb_keep_perfect_matching_never_breaks_the_perfect_matching():
    rng = np.random.default_rng(0)
    n = 8
    graph = staircase_graph(n)
    for _ in range(200):
        graph = perturb_keep_perfect_matching(graph, rng)
        assert max_matching_size(graph) == n
