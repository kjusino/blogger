import numpy as np
import pytest

from src.scaling import (
    rescaled_temperature,
    rescaled_magnetization,
    rescaled_susceptibility,
    binder_crossing,
    collapse_rmse,
)


def test_rescaled_temperature_zero_at_Tc():
    assert rescaled_temperature(2.269, 2.269, L=32, nu=1.0) == pytest.approx(0.0)


def test_rescaled_temperature_scales_with_L():
    x8 = rescaled_temperature(2.5, 2.269, L=8, nu=1.0)
    x32 = rescaled_temperature(2.5, 2.269, L=32, nu=1.0)
    assert x32 == pytest.approx(x8 * 4)


def test_rescaled_magnetization_and_susceptibility_formulas():
    m = np.array([0.5, 0.6])
    out = rescaled_magnetization(m, L=16, beta=0.125, nu=1.0)
    np.testing.assert_allclose(out, m * 16 ** 0.125)

    chi = np.array([2.0, 3.0])
    out_chi = rescaled_susceptibility(chi, L=16, gamma=1.75, nu=1.0)
    np.testing.assert_allclose(out_chi, chi * 16 ** (-1.75))


def test_binder_crossing_recovers_known_Tc_on_synthetic_data():
    """Construct synthetic Binder-cumulant curves that, by design, cross at a
    known T* for every adjacent pair of L, and check the estimator recovers
    T* (this isolates the crossing-detection logic from any Monte Carlo
    noise).
    """
    T_grid = np.linspace(2.0, 2.5, 200)
    T_star = 2.269
    L_values = [8, 16, 32]
    binder_by_L = {}
    for L in L_values:
        # A synthetic curve that decreases through T_star with an L-dependent
        # steepness (steeper for larger L, as real Binder curves are), but
        # all pass through the exact same value at T_star -- by construction
        # every pair crosses at T_star.
        steepness = 2.0 + 0.5 * np.log2(L)
        binder_by_L[L] = 0.6 - steepness * (T_grid - T_star)

    result = binder_crossing(T_grid, binder_by_L, L_values)
    assert result["Tc_estimate"] == pytest.approx(T_star, abs=0.01)
    assert len(result["crossings"]) == len(L_values) - 1


def test_binder_crossing_returns_none_when_curves_never_cross():
    T_grid = np.linspace(2.0, 2.5, 50)
    binder_by_L = {8: np.full(50, 0.5), 16: np.full(50, 0.6)}
    result = binder_crossing(T_grid, binder_by_L, [8, 16])
    assert result["Tc_estimate"] is None


def test_collapse_rmse_is_zero_for_identical_curves():
    x = {8: np.linspace(-1, 1, 20), 16: np.linspace(-1, 1, 20)}
    y = {8: np.sin(x[8]), 16: np.sin(x[16])}
    result = collapse_rmse(x, y, [8, 16])
    assert result["mean_rmse"] == pytest.approx(0.0, abs=1e-9)


def test_collapse_rmse_detects_genuine_mismatch():
    x = {8: np.linspace(-1, 1, 20), 16: np.linspace(-1, 1, 20)}
    y = {8: np.sin(x[8]), 16: np.sin(x[16]) + 0.5}
    result = collapse_rmse(x, y, [8, 16])
    assert result["mean_rmse"] > 0.1


def test_collapse_rmse_improves_after_correct_rescaling_of_synthetic_scaling_form():
    """Build data that exactly obeys the FSS ansatz m = L^-b * f(x), then
    check that RMSE computed on the rescaled curves is far smaller than RMSE
    computed on the raw (unrescaled) m vs T curves -- i.e. the rescaling
    procedure itself is doing real work, not just relabeling axes.
    """
    beta, nu, Tc = 0.125, 1.0, 2.269
    L_values = [8, 16, 32]
    T_by_L = {}
    m_by_L = {}
    x_by_L = {}
    y_by_L = {}
    for L in L_values:
        x = np.linspace(-2, 2, 100)
        f = np.exp(-0.5 * x ** 2)  # universal scaling function f(x)
        T = Tc + x / (L ** (1.0 / nu))
        m = f * L ** (-beta / nu)
        T_by_L[L] = T
        m_by_L[L] = m
        x_by_L[L] = x
        y_by_L[L] = m * (L ** (beta / nu))  # should recover f(x) for every L

    raw_rmse = collapse_rmse(T_by_L, m_by_L, L_values)["mean_rmse"]
    rescaled_rmse = collapse_rmse(x_by_L, y_by_L, L_values)["mean_rmse"]
    assert rescaled_rmse < raw_rmse
    assert rescaled_rmse < 1e-6
