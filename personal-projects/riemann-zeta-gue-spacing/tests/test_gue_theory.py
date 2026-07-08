import numpy as np
import pytest
from scipy.integrate import trapezoid

from src.gue_theory import (
    gue_surmise_pdf,
    gue_surmise_cdf,
    poisson_pdf,
    poisson_cdf,
    montgomery_pair_correlation,
)


def test_gue_cdf_boundary_values():
    assert gue_surmise_cdf(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
    assert gue_surmise_cdf(np.array([20.0]))[0] == pytest.approx(1.0, abs=1e-9)


def test_gue_cdf_matches_pdf_by_finite_difference():
    s = np.linspace(0.05, 3.0, 40)
    h = 1e-6
    numeric_derivative = (gue_surmise_cdf(s + h) - gue_surmise_cdf(s - h)) / (2 * h)
    analytic_pdf = gue_surmise_pdf(s)
    assert numeric_derivative == pytest.approx(analytic_pdf, rel=1e-4, abs=1e-6)


def test_gue_cdf_is_monotonic():
    s = np.linspace(0, 5, 200)
    cdf = gue_surmise_cdf(s)
    assert np.all(np.diff(cdf) >= -1e-12)


def test_gue_pdf_integrates_to_one():
    s = np.linspace(0, 30, 200_000)
    integral = trapezoid(gue_surmise_pdf(s), s)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_gue_pdf_mean_spacing_is_one():
    # By construction the Wigner surmise is normalized to unit mean spacing.
    s = np.linspace(0, 30, 200_000)
    mean = trapezoid(s * gue_surmise_pdf(s), s)
    assert mean == pytest.approx(1.0, abs=2e-3)


def test_poisson_pdf_cdf_consistency():
    s = np.linspace(0, 10, 500)
    assert poisson_cdf(s)[0] == pytest.approx(0.0)
    assert poisson_cdf(np.array([10.0]))[0] == pytest.approx(1.0, abs=1e-4)
    numeric = np.gradient(poisson_cdf(s), s)
    assert numeric == pytest.approx(poisson_pdf(s), rel=1e-2, abs=1e-3)


def test_montgomery_pair_correlation_limits():
    # R2(u) -> 0 as u -> 0 (perfect level repulsion in the continuum limit)
    assert montgomery_pair_correlation(np.array([0.0]))[0] == pytest.approx(0.0)
    # R2(u) -> 1 for large u (uncorrelated at long range)
    assert montgomery_pair_correlation(np.array([50.0]))[0] == pytest.approx(1.0, abs=1e-3)
    # R2 stays within [0, 1]
    u = np.linspace(0, 20, 500)
    r2 = montgomery_pair_correlation(u)
    assert np.all(r2 >= -1e-9)
    assert np.all(r2 <= 1.0 + 1e-9)


def test_montgomery_pair_correlation_first_zero_near_u_1():
    # 1 - (sin(pi u)/(pi u))^2 hits exactly 1 at every positive integer u,
    # since sin(pi u) = 0 there.
    u = np.array([1.0, 2.0, 3.0])
    r2 = montgomery_pair_correlation(u)
    assert r2 == pytest.approx(1.0, abs=1e-9)
