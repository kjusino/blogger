import numpy as np
import pytest

from pbs.quorum import ack_time, is_stale_fixed, is_stale_random, slowest_fixed_subset


def test_ack_time_is_wth_order_statistic():
    latencies = np.array(
        [
            [5.0, 1.0, 3.0, 2.0, 4.0],
            [10.0, 20.0, 30.0, 5.0, 1.0],
        ]
    )
    # W=1 -> minimum, W=5 -> maximum
    assert np.array_equal(ack_time(latencies, 1), np.array([1.0, 1.0]))
    assert np.array_equal(ack_time(latencies, 5), np.array([5.0, 30.0]))
    # W=3 -> third smallest
    assert np.array_equal(ack_time(latencies, 3), np.array([3.0, 10.0]))


def test_ack_time_rejects_out_of_range_w():
    latencies = np.zeros((2, 4))
    with pytest.raises(ValueError):
        ack_time(latencies, 0)
    with pytest.raises(ValueError):
        ack_time(latencies, 5)


def test_slowest_fixed_subset_picks_largest_multipliers():
    multipliers = np.array([1.0, 5.0, 0.5, 3.0, 2.0])
    idx = slowest_fixed_subset(multipliers, r=2)
    assert set(idx.tolist()) == {1, 3}  # multipliers 5.0 and 3.0


def test_is_stale_fixed_true_when_subset_all_behind_read_time():
    latencies = np.array([[1.0, 9.0, 9.0], [1.0, 9.0, 0.5]])
    read_time = np.array([5.0, 5.0])
    fixed_idx = np.array([1, 2])
    stale = is_stale_fixed(latencies, read_time, fixed_idx)
    assert stale.tolist() == [True, False]  # second trial: replica 2 has latency 0.5 < 5.0


def test_is_stale_random_never_stale_when_r_equals_n():
    # If the read quorum is the entire replica set, it always includes the
    # fastest replica, which by definition has latency <= ack_time <= read_time.
    rng = np.random.default_rng(42)
    latencies = rng.exponential(size=(1000, 4))
    ack = ack_time(latencies, 2)
    read_time = ack + 1.0
    stale = is_stale_random(latencies, read_time, r=4, rng=rng)
    assert not stale.any()


def test_is_stale_random_matches_manual_computation_small_case():
    rng = np.random.default_rng(7)
    latencies = np.array([[10.0, 20.0, 30.0]])
    read_time = np.array([25.0])
    # Force the "random" subset to be replicas [1, 2] (latencies 20, 30) via monkeypatched rng behavior:
    # instead, just check with r=1 that staleness requires the single sampled replica > 25.
    stale_count = 0
    trials = 5000
    latencies_rep = np.repeat(latencies, trials, axis=0)
    read_time_rep = np.repeat(read_time, trials, axis=0)
    stale = is_stale_random(latencies_rep, read_time_rep, r=1, rng=rng)
    # Exactly 1 of 3 replicas (the one with latency 30) is > 25, so P(stale) ~= 1/3
    assert np.isclose(stale.mean(), 1.0 / 3.0, atol=0.03)
