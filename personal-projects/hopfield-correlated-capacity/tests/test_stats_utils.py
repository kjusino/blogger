import numpy as np
import pytest

from src.stats_utils import (
    mean_and_sem,
    bootstrap_ci,
    find_critical_alpha,
    finite_size_extrapolate,
    fit_capacity_vs_rho,
)


def test_mean_and_sem():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, sem = mean_and_sem(values)
    assert mean == pytest.approx(3.0)
    assert sem > 0


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.5, scale=0.1, size=200)
    lo, hi = bootstrap_ci(values, rng, n_boot=1000)
    assert lo < np.mean(values) < hi


def test_find_critical_alpha_simple_step():
    alphas = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    overlaps = np.array([1.0, 1.0, 0.98, 0.5, 0.05])
    alpha_c = find_critical_alpha(alphas, overlaps, threshold=0.95)
    assert 0.15 < alpha_c < 0.20


def test_find_critical_alpha_never_crosses_returns_none():
    alphas = np.array([0.05, 0.10, 0.15])
    overlaps = np.array([1.0, 0.99, 0.98])
    assert find_critical_alpha(alphas, overlaps, threshold=0.95) is None


def test_finite_size_extrapolate_linear_data():
    # alpha_c(N) = 0.138 - 0.5/N  =>  intercept at 1/N=0 should be ~0.138
    ns = np.array([100.0, 200.0, 400.0])
    alpha_cs = 0.138 - 0.5 / ns
    fit = finite_size_extrapolate(ns, alpha_cs)
    assert fit.intercept == pytest.approx(0.138, abs=1e-6)


def test_fit_capacity_vs_rho_prefers_correct_model():
    rhos = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    alpha0 = 0.14
    # Data generated exactly from the linear ansatz -> linear fit should win.
    alpha_cs = alpha0 * (1 - rhos)
    fit = fit_capacity_vs_rho(rhos, alpha_cs)
    assert fit.better_model == "linear (H1)"
    assert fit.linear_rmse < 1e-6
    assert fit.spearman_rho < 0  # decreasing
