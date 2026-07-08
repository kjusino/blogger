import numpy as np
import pytest

from prime_bias.theory import fit_inverse_log_decay


def test_recovers_exact_linear_relationship():
    scales = np.array([1e4, 1e5, 1e6, 1e7, 1e8, 1e9], dtype=float)
    true_slope, true_intercept = 0.7, 0.0
    biases = true_slope / np.log(scales) + true_intercept
    fit = fit_inverse_log_decay(scales, biases)
    assert fit["slope"] == pytest.approx(true_slope, rel=1e-9)
    assert fit["intercept"] == pytest.approx(true_intercept, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_recovers_relationship_with_small_noise():
    rng = np.random.default_rng(0)
    scales = np.logspace(4, 9, 10)
    true_slope, true_intercept = 1.2, 0.01
    noise = rng.normal(0, 1e-4, size=scales.size)
    biases = true_slope / np.log(scales) + true_intercept + noise
    fit = fit_inverse_log_decay(scales, biases)
    assert fit["slope"] == pytest.approx(true_slope, rel=0.05)
    assert fit["r_squared"] > 0.99


def test_flat_bias_gives_near_zero_slope():
    scales = np.logspace(4, 9, 10)
    biases = np.full(scales.size, 0.05)
    fit = fit_inverse_log_decay(scales, biases)
    assert fit["slope"] == pytest.approx(0.0, abs=1e-9)
    assert fit["intercept"] == pytest.approx(0.05, abs=1e-9)
