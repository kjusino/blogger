import numpy as np
import pytest

from barren_plateaus import theory


def test_fit_exponential_recovers_known_slope():
    ns = np.array([2, 4, 6, 8, 10, 12])
    true_slope, true_intercept = -1.5, 3.0
    variances = 2.0 ** (true_slope * ns + true_intercept)
    fit = theory.fit_exponential(ns, variances)
    assert fit.slope == pytest.approx(true_slope, abs=1e-6)
    assert fit.intercept == pytest.approx(true_intercept, abs=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-8)


def test_fit_power_law_recovers_known_slope():
    ns = np.array([2, 4, 6, 8, 10, 12])
    true_slope, true_intercept = -1.2, 2.0
    variances = 2.0 ** (true_slope * np.log2(ns) + true_intercept)
    fit = theory.fit_power_law(ns, variances)
    assert fit.slope == pytest.approx(true_slope, abs=1e-6)
    assert fit.intercept == pytest.approx(true_intercept, abs=1e-6)
    assert fit.r2 == pytest.approx(1.0, abs=1e-8)


def test_exponential_model_beats_power_law_on_exponential_data():
    ns = np.array([2, 4, 6, 8, 10, 12, 14])
    variances = 2.0 ** (-1.8 * ns)
    exp_fit = theory.fit_exponential(ns, variances)
    pow_fit = theory.fit_power_law(ns, variances)
    assert exp_fit.r2 > pow_fit.r2


def test_power_law_model_beats_exponential_on_power_law_data():
    ns = np.array([2, 4, 6, 8, 10, 12, 14])
    variances = ns.astype(float) ** (-1.5)
    exp_fit = theory.fit_exponential(ns, variances)
    pow_fit = theory.fit_power_law(ns, variances)
    assert pow_fit.r2 > exp_fit.r2
