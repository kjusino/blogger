import numpy as np
import pytest

from hypercube_cutoff import chain as ch
from hypercube_cutoff import simulate as sim


def test_simulate_bitvector_trajectories_shape_and_bounds():
    rng = np.random.default_rng(0)
    n, t_max, trials = 12, 20, 50
    weights = sim.simulate_bitvector_trajectories(n, t_max, trials, rng)
    assert weights.shape == (t_max + 1, trials)
    assert np.all(weights[0] == 0)
    assert np.all((weights >= 0) & (weights <= n))


def test_simulate_weight_trajectories_shape_and_bounds():
    rng = np.random.default_rng(1)
    n, t_max, trials = 40, 30, 100
    weights = sim.simulate_weight_trajectories(n, t_max, trials, rng)
    assert weights.shape == (t_max + 1, trials)
    assert np.all(weights[0] == 0)
    assert np.all((weights >= 0) & (weights <= n))


def test_simulate_weight_trajectories_custom_start():
    rng = np.random.default_rng(2)
    n = 20
    weights = sim.simulate_weight_trajectories(n, 0, 10, rng, start_weight=7)
    assert np.all(weights[0] == 7)


def test_simulate_rejects_negative_t_max():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.simulate_bitvector_trajectories(10, -1, 5, rng)
    with pytest.raises(ValueError):
        sim.simulate_weight_trajectories(10, -1, 5, rng)


def test_simulate_weight_trajectories_rejects_bad_start_weight():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.simulate_weight_trajectories(10, 5, 5, rng, start_weight=-1)
    with pytest.raises(ValueError):
        sim.simulate_weight_trajectories(10, 5, 5, rng, start_weight=11)


def test_empirical_tv_distance_matches_exact_with_many_trials():
    rng = np.random.default_rng(3)
    n, t = 25, 60
    exact_tv = ch.tv_curve(n, [t])[0]
    weights = sim.simulate_weight_trajectories(n, t, 50000, rng)[-1]
    emp_tv, _ = sim.empirical_tv_distance(weights, n)
    assert emp_tv == pytest.approx(exact_tv, abs=0.02)


def test_empirical_tv_distance_ci_contains_point_estimate():
    rng = np.random.default_rng(4)
    n, t = 20, 40
    weights = sim.simulate_weight_trajectories(n, t, 5000, rng)[-1]
    tv, ci = sim.empirical_tv_distance(weights, n, n_bootstrap=200, rng=np.random.default_rng(5))
    assert ci[0] <= tv <= ci[1]


def test_empirical_tv_distance_requires_rng_for_bootstrap():
    weights = np.array([0, 1, 2, 3])
    with pytest.raises(ValueError):
        sim.empirical_tv_distance(weights, 5, n_bootstrap=10, rng=None)


def test_empirical_tv_distance_rejects_out_of_range_samples():
    weights = np.array([0, 1, 100])
    with pytest.raises(ValueError):
        sim.empirical_tv_distance(weights, 5)


def test_bitvector_and_weight_simulators_agree_in_law():
    """Both simulators encode the same Markov chain (via the exact lumping);
    their empirical TV estimates at the same (n, t) should agree within
    statistical error."""
    n, t, trials = 18, 25, 30000
    bv = sim.simulate_bitvector_trajectories(n, t, trials, np.random.default_rng(10))[-1]
    wv = sim.simulate_weight_trajectories(n, t, trials, np.random.default_rng(11))[-1]
    tv_bv, _ = sim.empirical_tv_distance(bv, n)
    tv_wv, _ = sim.empirical_tv_distance(wv, n)
    assert abs(tv_bv - tv_wv) < 0.03
