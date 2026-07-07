import numpy as np
import pytest

from src.experiment import (
    connected_fraction_curve,
    exact_connectivity_threshold,
    periodic_distance_matrix,
    ratio_summary,
    run_betti1_curve,
    run_threshold_experiment,
    sample_thresholds,
)
from src.graph import is_connected, rips_edges


def test_periodic_distance_matrix_symmetric_and_zero_diagonal():
    rng = np.random.default_rng(0)
    points = rng.random((6, 2))
    dist = periodic_distance_matrix(points)
    assert np.allclose(dist, dist.T)
    assert np.allclose(np.diag(dist), 0.0)


def test_exact_threshold_is_the_smallest_r_that_connects():
    rng = np.random.default_rng(42)
    points = rng.random((25, 2))
    threshold = exact_connectivity_threshold(points)

    # Just below the threshold, the graph must be disconnected;
    # at/just above it, connected. This is the defining property of the
    # longest-MST-edge = connectivity-threshold equivalence.
    assert not is_connected(25, rips_edges(points, threshold - 1e-9))
    assert is_connected(25, rips_edges(points, threshold + 1e-9))


def test_exact_threshold_trivial_for_zero_or_one_point():
    assert exact_connectivity_threshold(np.zeros((0, 2))) == 0.0
    assert exact_connectivity_threshold(np.array([[0.3, 0.3]])) == 0.0


def test_sample_thresholds_returns_one_value_per_trial():
    rng = np.random.default_rng(7)
    thresholds = sample_thresholds(n=20, trials=10, rng=rng)
    assert thresholds.shape == (10,)
    assert np.all(thresholds > 0)


def test_connected_fraction_curve_is_monotone_and_bounded():
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    r_grid = np.linspace(0, 1, 20)
    frac = connected_fraction_curve(thresholds, r_grid)
    assert np.all(frac >= 0) and np.all(frac <= 1)
    assert np.all(np.diff(frac) >= 0)
    assert frac[-1] == pytest.approx(1.0)
    assert frac[0] == pytest.approx(0.0)


def test_run_threshold_experiment_shape_and_theory_field():
    rng = np.random.default_rng(3)
    results = run_threshold_experiment(n_values=[30, 60], trials_per_n=8, rng=rng)
    assert set(results.keys()) == {30, 60}
    for n, result in results.items():
        assert result.thresholds.shape == (8,)
        assert result.theory_r_c > 0


def test_ratio_summary_is_order_one_for_moderate_n():
    # Weak sanity check (small n, small trial count => high variance): the
    # empirical/theoretical threshold ratio should be within a factor of ~3,
    # not, say, 100x off (which would indicate a real bug, not just noise).
    rng = np.random.default_rng(5)
    results = run_threshold_experiment(n_values=[80], trials_per_n=25, rng=rng)
    summary = ratio_summary(results[80])
    assert 0.3 < summary["ratio"] < 3.0


def test_run_betti1_curve_end_to_end_small_case():
    rng = np.random.default_rng(9)
    r_grid = np.linspace(0.05, 0.5, 5)
    result = run_betti1_curve(n=15, r_grid=r_grid, trials=3, rng=rng)
    assert result.mean_betti0.shape == (5,)
    assert result.mean_betti1.shape == (5,)
    # At the largest radius (near-complete graph), everything should be
    # connected (mean betti0 close to 1) and cycles filled in (betti1 low).
    assert result.mean_betti0[-1] < result.mean_betti0[0]
    assert np.all(result.mean_betti1 >= 0)
