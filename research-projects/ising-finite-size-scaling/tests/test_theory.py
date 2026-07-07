import numpy as np
import pytest
from scipy import integrate

from src import theory


def test_exact_Tc_value():
    assert theory.T_C == pytest.approx(2.0 / np.log(1.0 + np.sqrt(2.0)))
    assert theory.T_C == pytest.approx(2.269185314213022, abs=1e-12)


def test_elliptic_K_matches_independent_quadrature_away_from_singularity():
    for k in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        integrand = lambda theta, k=k: 1.0 / np.sqrt(1.0 - (k ** 2) * np.sin(theta) ** 2)
        reference, _ = integrate.quad(integrand, 0.0, np.pi / 2 - 1e-9)
        mine = theory._complete_elliptic_K(k)
        assert mine == pytest.approx(reference, rel=1e-5)


def test_elliptic_K_monotonic_increasing_in_k():
    ks = np.linspace(0.01, 0.99, 20)
    vals = [theory._complete_elliptic_K(k) for k in ks]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_onsager_energy_ground_state_limit():
    assert theory.onsager_energy(0.05) == pytest.approx(-2.0, abs=1e-6)


def test_onsager_energy_high_temperature_limit_near_zero():
    assert abs(theory.onsager_energy(1000.0)) < 0.01


def test_onsager_energy_at_Tc_matches_known_closed_form():
    # Onsager's exact result: u(Tc) = -J*sqrt(2).
    assert theory.onsager_energy(theory.T_C) == pytest.approx(-np.sqrt(2.0), abs=1e-8)


def test_onsager_energy_continuous_across_Tc():
    below = theory.onsager_energy(theory.T_C - 1e-5)
    above = theory.onsager_energy(theory.T_C + 1e-5)
    at = theory.onsager_energy(theory.T_C)
    assert below == pytest.approx(at, abs=1e-3)
    assert above == pytest.approx(at, abs=1e-3)


def test_onsager_energy_monotonic_in_T():
    Ts = np.linspace(0.5, 5.0, 30)
    vals = theory.onsager_energy(Ts)
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_onsager_energy_vectorized_matches_scalar_calls():
    Ts = np.array([0.8, 1.5, 2.0, 2.5, 4.0])
    vec = theory.onsager_energy(Ts)
    scalar = np.array([theory.onsager_energy(float(t)) for t in Ts])
    np.testing.assert_allclose(vec, scalar)


def test_onsager_magnetization_zero_above_Tc():
    assert theory.onsager_magnetization(theory.T_C + 0.5) == 0.0
    assert theory.onsager_magnetization(theory.T_C + 1e-6) == 0.0
    assert theory.onsager_magnetization(theory.T_C) == 0.0


def test_onsager_magnetization_near_one_at_low_T():
    assert theory.onsager_magnetization(0.1) == pytest.approx(1.0, abs=1e-6)


def test_onsager_magnetization_monotonic_decreasing_below_Tc():
    Ts = np.linspace(0.5, theory.T_C - 0.01, 20)
    vals = theory.onsager_magnetization(Ts)
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))


def test_onsager_magnetization_vectorized_matches_scalar_calls():
    Ts = np.array([0.5, 1.0, 1.8, 2.0, 2.269185314213022, 2.5, 3.0])
    vec = theory.onsager_magnetization(Ts)
    scalar = np.array([theory.onsager_magnetization(float(t)) for t in Ts])
    np.testing.assert_allclose(vec, scalar)


def test_critical_exponents_are_the_known_2d_ising_values():
    assert theory.BETA == pytest.approx(0.125)
    assert theory.GAMMA == pytest.approx(1.75)
    assert theory.NU == pytest.approx(1.0)
