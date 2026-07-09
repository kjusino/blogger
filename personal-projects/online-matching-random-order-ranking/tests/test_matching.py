import itertools

import numpy as np
import pytest

from src.graphs import (complete_bipartite, perfect_matching_graph,
                         random_bipartite, staircase_graph)
from src.matching import (BipartiteGraph, greedy_online_matching,
                           is_valid_matching, matching_size,
                           max_bipartite_matching, max_matching_size,
                           ranking_online_matching)


def brute_force_max_matching(graph):
    """Exhaustive search over all subsets of left vertices for the largest
    that admits a system of distinct representatives -- used only to check
    Hopcroft-Karp on small graphs (n <= 8), independent of its own code."""
    n_left, n_right = graph.n_left, graph.n_right
    adj = graph.adj_left
    best = 0
    for k in range(n_left, 0, -1):
        if k <= best:
            break
        for subset in itertools.combinations(range(n_left), k):
            candidate_neighbors = [adj[i] for i in subset]
            if _has_perfect_matching(candidate_neighbors, n_right):
                best = k
                break
    return best


def _has_perfect_matching(neighbor_lists, n_right):
    k = len(neighbor_lists)
    used = [False] * n_right

    def try_assign(idx, used_local):
        if idx == k:
            return True
        for v in neighbor_lists[idx]:
            if not used_local[v]:
                used_local[v] = True
                if try_assign(idx + 1, used_local):
                    return True
                used_local[v] = False
        return False

    return try_assign(0, used)


@pytest.mark.parametrize("seed", range(20))
def test_hopcroft_karp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n_left, n_right = rng.integers(1, 7), rng.integers(1, 7)
    p = rng.uniform(0.1, 0.7)
    edges = [(i, j) for i in range(n_left) for j in range(n_right)
             if rng.random() < p]
    graph = BipartiteGraph(int(n_left), int(n_right), edges)
    assert max_matching_size(graph) == brute_force_max_matching(graph)


def test_empty_graph_has_zero_matching():
    graph = BipartiteGraph(5, 5, [])
    assert max_matching_size(graph) == 0


def test_complete_bipartite_has_perfect_matching():
    graph = complete_bipartite(6)
    match_left, match_right, size = max_bipartite_matching(graph)
    assert size == 6
    assert is_valid_matching(graph, match_right)


def test_unique_perfect_matching_is_found():
    rng = np.random.default_rng(0)
    graph = perfect_matching_graph(7, rng)
    _, match_right, size = max_bipartite_matching(graph)
    assert size == 7
    for j, i in enumerate(match_right):
        assert (i, j) in graph.edges


def test_matching_size_counts_non_negative_one_entries():
    assert matching_size([0, -1, 2, -1, 4]) == 3
    assert matching_size([-1, -1]) == 0


def test_greedy_online_matching_is_always_valid():
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = int(rng.integers(2, 12))
        graph = random_bipartite(n, rng.uniform(0.1, 0.6), rng)
        order = list(rng.permutation(n))
        match = greedy_online_matching(graph, order)
        assert is_valid_matching(graph, match)


def test_ranking_online_matching_is_always_valid():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(2, 12))
        graph = random_bipartite(n, rng.uniform(0.1, 0.6), rng)
        rank = rng.permutation(n)
        order = list(rng.permutation(n))
        match = ranking_online_matching(graph, rank, order)
        assert is_valid_matching(graph, match)


def test_online_algorithms_never_beat_the_offline_optimum():
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(2, 16))
        graph = random_bipartite(n, rng.uniform(0.05, 0.5), rng)
        opt = max_matching_size(graph)
        order = list(rng.permutation(n))
        greedy_match = greedy_online_matching(graph, order)
        rank = rng.permutation(n)
        ranking_match = ranking_online_matching(graph, rank, order)
        assert matching_size(greedy_match) <= opt
        assert matching_size(ranking_match) <= opt


def test_ranking_with_identity_rank_and_order_matches_greedy_tiebreak():
    # With rank = identity, RANKING always prefers the lowest-index
    # available neighbor -- exactly greedy's fixed tie-break rule.
    graph = staircase_graph(6)
    order = list(range(6))
    rank = list(range(6))
    greedy_match = greedy_online_matching(graph, order)
    ranking_match = ranking_online_matching(graph, rank, order)
    assert greedy_match == ranking_match


def test_complete_bipartite_ranking_always_achieves_perfect_matching():
    # On K_{n,n}, RANKING (and greedy) can never fail to extend the
    # matching, regardless of rank or arrival order -- every online vertex
    # is adjacent to every unmatched offline vertex.
    rng = np.random.default_rng(4)
    graph = complete_bipartite(8)
    for _ in range(10):
        rank = rng.permutation(8)
        order = list(rng.permutation(8))
        match = ranking_online_matching(graph, rank, order)
        assert matching_size(match) == 8


def test_with_edge_toggled_round_trips():
    graph = BipartiteGraph(3, 3, [(0, 0), (1, 1)])
    toggled_on = graph.with_edge_toggled(2, 2)
    assert (2, 2) in toggled_on.edges
    toggled_off = toggled_on.with_edge_toggled(2, 2)
    assert toggled_off.edges == graph.edges


def test_out_of_range_edge_raises():
    with pytest.raises(ValueError):
        BipartiteGraph(2, 2, [(0, 5)])
