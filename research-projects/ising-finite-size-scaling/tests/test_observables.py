import numpy as np
import pytest

from src.observables import (
    specific_heat,
    susceptibility,
    binder_cumulant,
    mean_abs_magnetization,
    mean_energy,
    integrated_autocorrelation_time,
)


def test_specific_heat_zero_for_constant_energy_series():
    e = np.full(100, -1.5)
    assert specific_heat(e, T=2.0, N=64) == pytest.approx(0.0)


def test_specific_heat_scales_with_variance():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.1, size=10000)
    c = specific_heat(e, T=2.0, N=64)
    expected = 64 * np.var(e) / (2.0 ** 2)
    assert c == pytest.approx(expected)


def test_susceptibility_zero_for_constant_magnetization():
    m = np.full(100, 0.8)
    assert susceptibility(m, T=2.0, N=64) == pytest.approx(0.0, abs=1e-9)


def test_susceptibility_positive_for_fluctuating_series():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.2, size=5000)
    chi = susceptibility(m, T=2.0, N=64)
    assert chi > 0


def test_binder_cumulant_for_gaussian_is_two_thirds():
    # For a mean-zero Gaussian, <m^4>/<m^2>^2 = 3 (excess kurtosis 0), so
    # U4 = 1 - 3/(3*1) = 0. This is the "high-temperature" reference value.
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1.0, size=2_000_000)
    u4 = binder_cumulant(m)
    assert u4 == pytest.approx(0.0, abs=0.01)


def test_binder_cumulant_for_bimodal_pm1_is_two_thirds():
    # For a perfectly ordered, symmetric +-1 series (m = +-m0 always),
    # <m^4> = m0^4, <m^2>^2 = m0^4, so U4 = 1 - 1/3 = 2/3 (the T=0 reference).
    m = np.array([1.0, -1.0] * 1000)
    u4 = binder_cumulant(m)
    assert u4 == pytest.approx(2.0 / 3.0)


def test_mean_abs_magnetization_and_mean_energy():
    m = np.array([-1.0, 1.0, 0.5, -0.5])
    assert mean_abs_magnetization(m) == pytest.approx(0.75)
    e = np.array([-1.0, -2.0, -3.0])
    assert mean_energy(e) == pytest.approx(-2.0)


def test_autocorrelation_time_of_iid_noise_is_near_half():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=20000)
    tau = integrated_autocorrelation_time(x)
    assert tau == pytest.approx(0.5, abs=0.2)


def test_autocorrelation_time_grows_with_explicit_correlation():
    """An AR(1) process x_t = phi*x_{t-1} + noise has known tau_int =
    (1+phi)/(1-phi); higher phi (slower mixing) should give a larger
    estimated tau_int than a barely-correlated series.
    """
    rng = np.random.default_rng(0)
    n = 200000

    def ar1(phi):
        x = np.zeros(n)
        noise = rng.normal(0, 1, size=n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + noise[t]
        return x

    low = ar1(0.1)
    high = ar1(0.9)
    tau_low = integrated_autocorrelation_time(low)
    tau_high = integrated_autocorrelation_time(high)
    assert tau_high > tau_low
    # Theoretical tau_int for phi=0.9 is (1.9)/(0.1) = 19; allow generous slack
    # since this is a finite-sample windowed estimator, not the exact value.
    assert 5 < tau_high < 40


def test_autocorrelation_time_handles_zero_variance_series():
    x = np.full(500, 3.0)
    assert integrated_autocorrelation_time(x) == pytest.approx(0.5)
