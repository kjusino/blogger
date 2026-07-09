import numpy as np
import pytest

from src.fitting import kkl_scaling_ratio, power_law_fit


def test_power_law_fit_recovers_known_exponent_noiseless():
    xs = np.geomspace(10, 10000, 40)
    ys = 3.0 * xs**0.5
    fit = power_law_fit(xs, ys)
    assert fit.exponent == pytest.approx(0.5, abs=1e-6)
    assert fit.r_squared > 0.999


def test_power_law_fit_recovers_known_exponent_with_noise():
    rng = np.random.default_rng(0)
    xs = np.geomspace(10, 10000, 200)
    ys = 2.0 * xs**1.3 * np.exp(rng.normal(0, 0.05, size=xs.shape))
    fit = power_law_fit(xs, ys)
    assert fit.exponent == pytest.approx(1.3, abs=0.02)
    lo, hi = fit.exponent_ci95()
    assert lo < 1.3 < hi


def test_power_law_fit_rejects_nonpositive_input():
    with pytest.raises(ValueError):
        power_law_fit(np.array([1, 2, -3]), np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        power_law_fit(np.array([1, 2, 3]), np.array([1, 0, 3]))


def test_power_law_fit_requires_at_least_three_points():
    with pytest.raises(ValueError):
        power_law_fit(np.array([1, 2]), np.array([1, 2]))


def test_kkl_scaling_ratio_positive_for_valid_function():
    # Tribes-like instance: small max influence but ratio should still be positive.
    ratio = kkl_scaling_ratio(max_influence=0.01, variance=1.0, n=64)
    assert ratio > 0


def test_kkl_scaling_ratio_vacuous_for_constant_function():
    ratio = kkl_scaling_ratio(max_influence=0.0, variance=0.0, n=10)
    assert ratio == float("inf")


def test_kkl_scaling_ratio_rejects_tiny_n():
    with pytest.raises(ValueError):
        kkl_scaling_ratio(max_influence=0.1, variance=1.0, n=1)
