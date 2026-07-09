import numpy as np

from src.stats_utils import fit_power_law


def test_fit_power_law_recovers_known_exponent_noiseless():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    c_true, a_true = 3.0, -0.5
    y = c_true * x ** a_true
    fit = fit_power_law(x, y)
    assert abs(fit["exponent"] - a_true) < 1e-8
    assert abs(fit["c"] - c_true) < 1e-6
    assert fit["r_squared"] > 0.999
    assert fit["n"] == 7


def test_fit_power_law_ci_contains_true_exponent_under_small_noise():
    rng = np.random.default_rng(0)
    x = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    a_true, c_true = -0.5, 2.0
    noise = rng.normal(scale=0.02, size=len(x))
    y = c_true * x ** a_true * np.exp(noise)
    fit = fit_power_law(x, y)
    lo, hi = fit["exponent_ci95"]
    assert lo < a_true < hi


def test_fit_power_law_handles_insufficient_or_invalid_data():
    fit = fit_power_law([1.0, 2.0], [1.0, 0.5])
    assert np.isnan(fit["exponent"])
    assert fit["n"] == 2

    fit2 = fit_power_law([1.0, -2.0, 3.0, 4.0], [1.0, 2.0, np.nan, 0.5])
    assert fit2["n"] == 2
