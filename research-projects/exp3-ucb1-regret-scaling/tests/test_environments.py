import numpy as np
import pytest

from src.environments import StochasticBernoulliEnv, SwitchingBernoulliEnv


def test_stochastic_reward_table_is_binary():
    rng = np.random.default_rng(0)
    env = StochasticBernoulliEnv([0.2, 0.5, 0.8], rng)
    table = env.reward_table(1000)
    assert table.shape == (1000, 3)
    assert set(np.unique(table)).issubset({0.0, 1.0})


def test_stochastic_reward_table_matches_means():
    rng = np.random.default_rng(0)
    means = [0.1, 0.5, 0.9]
    env = StochasticBernoulliEnv(means, rng)
    table = env.reward_table(50000)
    empirical = table.mean(axis=0)
    assert np.allclose(empirical, means, atol=0.01)


def test_stochastic_env_rejects_out_of_range_means():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        StochasticBernoulliEnv([0.5, 1.2], rng)


def test_switching_schedule_cycles_through_all_arms():
    rng = np.random.default_rng(0)
    env = SwitchingBernoulliEnv(n_arms=4, segment_length=10, delta=0.3, rng=rng)
    schedule = env.good_arm_schedule(horizon=40)
    assert schedule[:10].tolist() == [0] * 10
    assert schedule[10:20].tolist() == [1] * 10
    assert schedule[20:30].tolist() == [2] * 10
    assert schedule[30:40].tolist() == [3] * 10


def test_switching_env_rejects_bad_params():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        SwitchingBernoulliEnv(n_arms=1, segment_length=10, delta=0.3, rng=rng)
    with pytest.raises(ValueError):
        SwitchingBernoulliEnv(n_arms=3, segment_length=10, delta=0.0, rng=rng)
    with pytest.raises(ValueError):
        SwitchingBernoulliEnv(n_arms=3, segment_length=0, delta=0.3, rng=rng)


def test_switching_env_good_arm_has_higher_mean_within_segment():
    rng = np.random.default_rng(0)
    env = SwitchingBernoulliEnv(n_arms=3, segment_length=20000, delta=0.4, rng=rng)
    table = env.reward_table(horizon=20000)  # exactly one full segment, arm 0 is good
    empirical = table.mean(axis=0)
    assert empirical[0] > empirical[1] + 0.1
    assert empirical[0] > empirical[2] + 0.1
    assert abs(empirical[0] - 0.7) < 0.02


def test_switching_env_is_symmetric_across_a_full_cycle():
    # Every arm is "good" for exactly one segment per cycle, so total
    # reward per arm over a full cycle should be (statistically) equal.
    rng = np.random.default_rng(0)
    n_arms = 4
    segment_length = 5000
    env = SwitchingBernoulliEnv(n_arms=n_arms, segment_length=segment_length, delta=0.3, rng=rng)
    table = env.reward_table(horizon=segment_length * n_arms)
    totals = table.sum(axis=0)
    assert np.std(totals) / np.mean(totals) < 0.02
