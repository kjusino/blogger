import math

import numpy as np
import pytest

from src.bandits import EXP3, UCB1


def test_exp3_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    algo = EXP3(5, horizon=1000, rng=rng)
    probs = algo.probabilities()
    assert probs.shape == (5,)
    assert math.isclose(probs.sum(), 1.0, rel_tol=1e-12)


def test_exp3_probabilities_respect_exploration_floor():
    rng = np.random.default_rng(0)
    algo = EXP3(5, horizon=1000, rng=rng)
    for _ in range(50):
        arm = algo.select_arm()
        probs = algo._last_probs
        assert np.all(probs >= algo.gamma / algo.k - 1e-12)
        algo.update(arm, reward=rng.random())


def test_exp3_full_exploration_is_uniform_in_expectation():
    # gamma = 1 means every weight update becomes irrelevant to the mix:
    # probabilities() collapses to exactly uniform regardless of history.
    rng = np.random.default_rng(1)
    algo = EXP3(4, horizon=10, rng=rng, gamma=1.0)
    counts = np.zeros(4)
    n_rounds = 20000
    for _ in range(n_rounds):
        arm = algo.select_arm()
        counts[arm] += 1
        algo.update(arm, reward=1.0 if arm == 0 else 0.0)
    frequencies = counts / n_rounds
    assert np.allclose(frequencies, 0.25, atol=0.02)


def test_exp3_rejects_invalid_gamma():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        EXP3(3, horizon=10, rng=rng, gamma=0.0)
    with pytest.raises(ValueError):
        EXP3(3, horizon=10, rng=rng, gamma=1.5)


def test_exp3_rejects_out_of_range_reward():
    rng = np.random.default_rng(0)
    algo = EXP3(3, horizon=10, rng=rng)
    arm = algo.select_arm()
    with pytest.raises(ValueError):
        algo.update(arm, reward=1.5)


def test_exp3_update_must_match_selected_arm():
    rng = np.random.default_rng(0)
    algo = EXP3(3, horizon=10, rng=rng)
    arm = algo.select_arm()
    other = (arm + 1) % 3
    with pytest.raises(ValueError):
        algo.update(other, reward=0.5)


def test_exp3_learns_to_favor_obviously_best_arm():
    # Deterministic, stationary rewards: arm 0 always pays 1, others 0.
    rng = np.random.default_rng(2)
    horizon = 3000
    algo = EXP3(4, horizon=horizon, rng=rng)
    counts = np.zeros(4)
    for _ in range(horizon):
        arm = algo.select_arm()
        counts[arm] += 1
        algo.update(arm, reward=1.0 if arm == 0 else 0.0)
    # Should play the best arm far more than the 1/K uniform baseline,
    # even though the gamma/K exploration floor keeps it from reaching 100%.
    assert counts[0] / horizon > 2.0 / 4


def test_ucb1_plays_every_arm_once_before_exploiting():
    algo = UCB1(5)
    played_first_round = []
    for _ in range(5):
        arm = algo.select_arm()
        played_first_round.append(arm)
        algo.update(arm, reward=0.5)
    assert sorted(played_first_round) == [0, 1, 2, 3, 4]


def test_ucb1_rejects_single_arm():
    with pytest.raises(ValueError):
        UCB1(1)


def test_ucb1_update_must_match_selected_arm():
    algo = UCB1(3)
    arm = algo.select_arm()
    other = (arm + 1) % 3
    with pytest.raises(ValueError):
        algo.update(other, reward=0.5)


def test_ucb1_converges_to_obviously_best_arm():
    algo = UCB1(4)
    horizon = 3000
    counts = np.zeros(4)
    for _ in range(horizon):
        arm = algo.select_arm()
        counts[arm] += 1
        algo.update(arm, reward=1.0 if arm == 0 else 0.0)
    assert counts[0] / horizon > 0.9
