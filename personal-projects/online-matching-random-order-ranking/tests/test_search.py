import numpy as np

from src.graphs import complete_bipartite, staircase_graph
from src.matching import max_matching_size
from src.search import (ONE_MINUS_INV_E, exact_finite_floor, greedy_ratio,
                         greedy_ratios_random_orders, mean_ranking_ratio,
                         ranking_ratio_single, ranking_ratios,
                         search_worst_case_graph)


def test_exact_finite_floor_converges_to_one_minus_inv_e_from_above():
    prev = 1.0
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 1024]:
        floor = exact_finite_floor(n)
        assert floor > ONE_MINUS_INV_E
        assert floor < prev  # strictly decreasing towards the limit
        prev = floor
    assert abs(exact_finite_floor(100000) - ONE_MINUS_INV_E) < 1e-3


def test_ranking_ratio_is_always_one_on_complete_bipartite():
    rng = np.random.default_rng(0)
    graph = complete_bipartite(10)
    ratios = ranking_ratios(graph, list(range(10)), rng, 25, opt_size=10)
    assert all(r == 1.0 for r in ratios)


def test_ranking_ratios_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    graph = staircase_graph(12)
    ratios = ranking_ratios(graph, list(range(12)), rng, 50)
    assert all(0.0 <= r <= 1.0 for r in ratios)


def test_greedy_ratio_is_one_on_complete_bipartite():
    graph = complete_bipartite(6)
    assert greedy_ratio(graph, list(range(6)), opt_size=6) == 1.0


def test_greedy_ratios_random_orders_bounded():
    rng = np.random.default_rng(2)
    graph = staircase_graph(9)
    ratios = greedy_ratios_random_orders(graph, rng, 30)
    assert all(0.0 <= r <= 1.0 for r in ratios)
    assert len(ratios) == 30


def test_ranking_ratio_single_matches_batch_statistics():
    rng_batch = np.random.default_rng(42)
    rng_single = np.random.default_rng(42)
    graph = staircase_graph(10)
    order = list(range(10))
    batch = ranking_ratios(graph, order, rng_batch, 5, opt_size=10)
    single = [ranking_ratio_single(graph, order, rng_single, 10) for _ in range(5)]
    assert batch == single  # identical seeds -> identical draws


def test_search_history_is_monotone_non_increasing():
    rng = np.random.default_rng(0)
    n = 8
    _, best_score, history = search_worst_case_graph(
        n, list(range(n)), rng, n_iterations=40, trials_per_eval=15)
    assert all(history[i] >= history[i + 1] for i in range(len(history) - 1))
    assert history[-1] == best_score


def test_search_finds_a_graph_at_least_as_hard_as_the_starting_point():
    rng = np.random.default_rng(0)
    n = 8
    order = list(range(n))
    start = staircase_graph(n)
    start_score, _ = mean_ranking_ratio(start, order, rng, 60, opt_size=n)
    best_graph, best_score, _ = search_worst_case_graph(
        n, order, rng, n_iterations=60, trials_per_eval=20, init_graph=start)
    assert best_score <= start_score + 1e-9
    assert max_matching_size(best_graph) == n


def test_search_never_drives_mean_ratio_below_the_asymptotic_floor():
    # A stronger, large-sample re-check: whatever graph the search settles
    # on, the *mean* ratio over many fresh trials should not fall below the
    # theorem's floor (individual trials may, the mean should not).
    rng = np.random.default_rng(7)
    n = 12
    order = list(range(n))
    best_graph, _, _ = search_worst_case_graph(
        n, order, rng, n_iterations=80, trials_per_eval=25)
    mean_ratio, _ = mean_ranking_ratio(best_graph, order, rng, 400, opt_size=n)
    assert mean_ratio >= ONE_MINUS_INV_E - 0.05
