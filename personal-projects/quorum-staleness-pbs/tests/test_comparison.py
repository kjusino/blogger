import numpy as np

from pbs.comparison import area_between, monte_carlo_noise_floor, sup_distance


def test_sup_distance_identical_curves_zero():
    curve = np.array([0.1, 0.2, 0.3])
    assert sup_distance(curve, curve) == 0.0


def test_sup_distance_picks_max_gap():
    a = np.array([0.1, 0.5, 0.9])
    b = np.array([0.15, 0.2, 0.85])
    assert np.isclose(sup_distance(a, b), 0.3)


def test_area_between_zero_for_identical_curves():
    deltas = np.linspace(0, 1, 11)
    curve = np.linspace(1, 0, 11)
    assert area_between(curve, curve, deltas) == 0.0


def test_area_between_constant_gap_equals_gap_times_range():
    deltas = np.linspace(0, 2, 3)  # [0, 1, 2]
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.3, 0.3, 0.3])
    assert np.isclose(area_between(a, b, deltas), 0.2 * 2)


def test_monte_carlo_noise_floor_shrinks_with_more_trials():
    small = monte_carlo_noise_floor(1_000)
    large = monte_carlo_noise_floor(100_000)
    assert large < small
    assert small > 0
