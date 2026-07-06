import random

import numpy as np
import pytest

from src.experiment import (
    fit_logistic,
    fit_power_law,
    measure_bler,
    measure_bler_curve,
    waterfall_width,
)


def test_fit_logistic_recovers_known_parameters():
    true_eps50, true_k = 0.43, 150.0
    eps = np.linspace(0.35, 0.51, 25)
    bler = 1.0 / (1.0 + np.exp(-true_k * (eps - true_eps50)))
    eps50, k = fit_logistic(eps.tolist(), bler.tolist())
    assert eps50 == pytest.approx(true_eps50, abs=1e-3)
    assert k == pytest.approx(true_k, rel=0.05)


def test_fit_logistic_handles_degenerate_all_zero_data():
    eps = [0.1, 0.2, 0.3]
    bler = [0.0, 0.0, 0.0]
    eps50, k = fit_logistic(eps, bler)
    assert k > 0  # must not crash or return a non-positive steepness


def test_waterfall_width_positive_and_shrinks_with_steepness():
    assert waterfall_width(100) > waterfall_width(1000) > 0


def test_fit_power_law_recovers_known_exponent():
    ns = [128, 256, 512, 1024, 2048]
    true_c, true_p = 0.6, -2 / 3
    ys = [true_c * n**true_p for n in ns]
    c, p = fit_power_law(ns, ys)
    assert p == pytest.approx(true_p, abs=1e-6)
    assert c == pytest.approx(true_c, rel=1e-6)


def test_fit_power_law_requires_two_positive_points():
    with pytest.raises(ValueError):
        fit_power_law([128, 256], [0.0, 0.0])


def test_measure_bler_is_zero_deep_below_threshold():
    rng = random.Random(0)
    bler = measure_bler(500, 3, 6, epsilon=0.05, n_graph_instances=2, trials_per_graph=30, rng=rng)
    assert bler < 0.05


def test_measure_bler_is_near_one_deep_above_threshold():
    rng = random.Random(0)
    bler = measure_bler(500, 3, 6, epsilon=0.8, n_graph_instances=2, trials_per_graph=30, rng=rng)
    assert bler > 0.9


def test_measure_bler_curve_produces_monotone_trend_and_sane_fit():
    rng = random.Random(42)
    curve = measure_bler_curve(
        300,
        3,
        6,
        eps_star=0.4294,
        rng=rng,
        n_graph_instances=2,
        coarse_trials=15,
        fine_trials=25,
        n_coarse=5,
        n_fine=5,
    )
    assert len(curve.epsilons) == 10
    assert 0.0 < curve.eps50 < 1.0
    assert curve.steepness > 0
    assert curve.width_90_10 > 0
    # BLER should broadly increase with epsilon across the sampled points
    order = sorted(range(len(curve.epsilons)), key=lambda i: curve.epsilons[i])
    sorted_blers = [curve.blers[i] for i in order]
    assert sorted_blers[-1] >= sorted_blers[0]
