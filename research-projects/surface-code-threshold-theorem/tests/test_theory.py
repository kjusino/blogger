import numpy as np
import pytest

from src.theory import (
    estimate_threshold,
    find_pairwise_crossing,
    fit_subthreshold_exponent,
    predicted_subthreshold_exponent,
)


@pytest.mark.parametrize(
    "distance,expected",
    [(3, 2), (5, 3), (7, 4), (9, 5), (11, 6)],
)
def test_predicted_subthreshold_exponent(distance, expected):
    assert predicted_subthreshold_exponent(distance) == expected


def test_predicted_subthreshold_exponent_rejects_even_distance():
    with pytest.raises(ValueError):
        predicted_subthreshold_exponent(4)


def test_fit_recovers_known_synthetic_power_law():
    p = np.geomspace(1e-4, 1e-2, 12)
    true_slope = 3.0
    rates = 0.5 * p ** true_slope  # P_L = A * p^k with A=0.5, k=3
    fit = fit_subthreshold_exponent(p, rates, distance=5)
    assert fit.slope == pytest.approx(true_slope, abs=1e-6)
    assert fit.r_value == pytest.approx(1.0, abs=1e-9)
    assert fit.predicted_slope == 3


def test_fit_requires_at_least_two_nonzero_points():
    p = np.array([0.01, 0.02])
    rates = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        fit_subthreshold_exponent(p, rates, distance=3)


def test_find_pairwise_crossing_recovers_known_intersection():
    # Two power laws p^2 and 2*p^3 cross where p^2 = 2 p^3 -> p = 0.5.
    p = np.geomspace(0.05, 5.0, 400)
    low_d = p ** 2
    high_d = 2 * p ** 3
    crossing = find_pairwise_crossing(p, low_d, high_d)
    assert crossing == pytest.approx(0.5, rel=0.05)


def test_find_pairwise_crossing_returns_none_when_curves_never_cross():
    p = np.geomspace(1e-4, 1e-2, 50)
    low_d = p ** 2
    high_d = p ** 3  # always below low_d in this range (p < 1), never crosses
    assert find_pairwise_crossing(p, low_d, high_d) is None


def test_estimate_threshold_on_synthetic_three_distance_family():
    # Construct P_L(d, p) = (p / p_th)^floor((d+1)/2) for p_th = 0.01,
    # which by construction crosses at exactly p = p_th for every pair.
    p_th_true = 0.01
    p = np.geomspace(0.001, 0.1, 40)
    rates_by_distance = {
        d: (p / p_th_true) ** ((d + 1) // 2) for d in (3, 5, 7)
    }
    estimate, crossings = estimate_threshold(p, rates_by_distance)
    assert estimate == pytest.approx(p_th_true, rel=0.02)
    for pair_estimate in crossings.values():
        assert pair_estimate == pytest.approx(p_th_true, rel=0.02)
