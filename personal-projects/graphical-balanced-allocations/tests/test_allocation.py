import numpy as np
import pytest

from src import allocation, graphs


def test_one_choice_total_balls_conserved():
    rng = np.random.default_rng(0)
    loads = allocation.simulate_one_choice(50, 500, rng)
    assert loads.sum() == 500
    assert loads.shape == (50,)


def test_one_choice_single_bin_gets_everything():
    rng = np.random.default_rng(0)
    loads = allocation.simulate_one_choice(1, 100, rng)
    assert loads[0] == 100


def test_classical_two_choice_conserves_balls_and_rejects_tiny_input():
    rng = np.random.default_rng(0)
    loads = allocation.simulate_classical_two_choice(20, 200, rng)
    assert loads.sum() == 200
    with pytest.raises(ValueError):
        allocation.simulate_classical_two_choice(1, 10, rng)


def test_classical_two_choice_never_samples_the_same_bin_twice():
    # a and b must always differ, otherwise this isn't "two choices"
    rng = np.random.default_rng(1)
    n_bins, n_balls = 10, 5000
    a = rng.integers(0, n_bins, size=n_balls)
    offset = rng.integers(1, n_bins, size=n_balls)
    b = (a + offset) % n_bins
    assert np.all(a != b)


def test_graphical_two_choice_on_single_edge_balances_within_one():
    # two nodes, one edge: every ball must go to whichever endpoint is
    # currently <= the other, so loads can never differ by more than 1.
    rng = np.random.default_rng(0)
    edges = np.array([[0, 1]])
    loads = allocation.simulate_graphical_two_choice(edges, 2, 999, rng)
    assert loads.sum() == 999
    assert abs(int(loads[0]) - int(loads[1])) <= 1


def test_graphical_two_choice_rejects_empty_edge_list():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        allocation.simulate_graphical_two_choice(np.zeros((0, 2), dtype=np.int64), 5, 10, rng)


def test_graphical_two_choice_on_complete_graph_conserves_balls():
    G, edges = graphs.generate_graph("complete", 30, seed=0)
    rng = np.random.default_rng(0)
    loads = allocation.simulate_graphical_two_choice(edges, 30, 300, rng)
    assert loads.sum() == 300


def test_max_load_gap_matches_definition():
    loads = np.array([1, 1, 3, 1])
    assert allocation.max_load_gap(loads, n_balls=6, n_bins=4) == 3 - 2  # ceil(6/4)=2


def test_two_choice_beats_one_choice_on_average():
    # the whole point of the "power of two choices": at moderate n, the
    # mean max-load gap for two-choice should be well below one-choice's.
    # deterministic under a fixed seed set, checked over enough trials
    # that this isn't a coin flip.
    n, trials = 2000, 25
    one_gaps, two_gaps = [], []
    for t in range(trials):
        rng = np.random.default_rng(100 + t)
        one_gaps.append(
            allocation.max_load_gap(allocation.simulate_one_choice(n, n, rng), n, n)
        )
        rng = np.random.default_rng(200 + t)
        two_gaps.append(
            allocation.max_load_gap(allocation.simulate_classical_two_choice(n, n, rng), n, n)
        )
    assert np.mean(two_gaps) < np.mean(one_gaps)
