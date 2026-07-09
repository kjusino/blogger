import numpy as np
import pytest

from src.rate_fitting import fit_asymptotic_rate


def test_recovers_exact_rate_from_noiseless_geometric_sequence():
    rate_true = 0.87
    k = np.arange(300)
    hist = 10.0 * rate_true**k
    fit = fit_asymptotic_rate(hist)
    assert fit.rate == pytest.approx(rate_true, rel=1e-6)
    assert fit.r_squared > 0.999


def test_recovers_approximate_rate_under_small_multiplicative_noise():
    rng = np.random.default_rng(0)
    rate_true = 0.75
    k = np.arange(500)
    noise = np.exp(rng.normal(0, 0.02, size=k.shape))
    hist = 5.0 * rate_true**k * noise
    fit = fit_asymptotic_rate(hist)
    assert fit.rate == pytest.approx(rate_true, rel=0.02)
    assert fit.r_squared > 0.98


def test_handles_history_with_trailing_zeros_from_underflow():
    rate_true = 0.5
    k = np.arange(120)
    hist = rate_true**k
    hist[80:] = 0.0  # simulate hitting float64 underflow / exact-zero residual
    fit = fit_asymptotic_rate(hist)
    assert fit.rate == pytest.approx(rate_true, rel=1e-6)


def test_returns_nan_for_too_short_history():
    fit = fit_asymptotic_rate(np.array([1.0, 0.5]), min_points=8)
    assert np.isnan(fit.rate)
    assert np.isnan(fit.r_squared)


def test_returns_nan_for_all_zero_history():
    fit = fit_asymptotic_rate(np.zeros(50))
    assert np.isnan(fit.rate)


def test_faster_decay_gives_smaller_fitted_rate():
    k = np.arange(200)
    fit_fast = fit_asymptotic_rate(0.3**k)
    fit_slow = fit_asymptotic_rate(0.95**k)
    assert fit_fast.rate < fit_slow.rate
