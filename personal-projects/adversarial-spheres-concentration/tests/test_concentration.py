import numpy as np
import pytest

from src.concentration import (
    cap_measure,
    cap_angle_for_measure,
    chord_from_geodesic,
    levy_ceiling_exact,
    levy_bound_asymptotic,
)


def test_cap_measure_hemisphere_is_half():
    for d in [2, 3, 5, 10, 100]:
        assert np.isclose(cap_measure(np.pi / 2, d), 0.5, atol=1e-9)


def test_cap_measure_full_sphere_is_one():
    for d in [3, 10]:
        assert np.isclose(cap_measure(np.pi, d), 1.0, atol=1e-9)


def test_cap_measure_empty_is_zero():
    for d in [3, 10]:
        assert np.isclose(cap_measure(0.0, d), 0.0, atol=1e-9)


def test_cap_measure_monotonic_in_angle():
    d = 12
    thetas = np.linspace(0, np.pi, 50)
    vals = cap_measure(thetas, d)
    assert np.all(np.diff(vals) >= -1e-12)


def test_cap_measure_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for d in [3, 8, 30]:
        for theta_deg in [40, 90, 130]:
            theta = np.radians(theta_deg)
            n = 60000
            x = rng.normal(size=(n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            e1 = np.zeros(d)
            e1[0] = 1.0
            frac = np.mean((x @ e1) >= np.cos(theta))
            formula = cap_measure(theta, d)
            # Monte Carlo standard error ~ sqrt(p(1-p)/n); allow generous tolerance
            assert abs(frac - formula) < 0.01


def test_cap_angle_for_measure_is_inverse_of_cap_measure():
    d = 15
    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        theta = cap_angle_for_measure(p, d)
        assert np.isclose(cap_measure(theta, d), p, atol=1e-6)


def test_chord_from_geodesic_zero_and_antipode():
    assert np.isclose(chord_from_geodesic(2.0, 0.0), 0.0)
    assert np.isclose(chord_from_geodesic(2.0, np.pi), 4.0)


def test_levy_ceiling_decreases_with_dimension():
    # higher dimension -> stronger concentration -> smaller robustness ceiling
    ceilings = [levy_ceiling_exact(p_minor=0.4, d=d, radius=1.0) for d in [4, 16, 64, 256]]
    assert all(ceilings[i] > ceilings[i + 1] for i in range(len(ceilings) - 1))


def test_levy_ceiling_shrinks_roughly_as_inverse_sqrt_d():
    d1, d2 = 50, 200
    c1 = levy_ceiling_exact(p_minor=0.4, d=d1, radius=1.0)
    c2 = levy_ceiling_exact(p_minor=0.4, d=d2, radius=1.0)
    ratio = c1 / c2
    expected_ratio = np.sqrt(d2 / d1)
    assert 0.5 * expected_ratio < ratio < 1.5 * expected_ratio


def test_levy_ceiling_degenerate_minority_is_infinite():
    assert levy_ceiling_exact(p_minor=0.0, d=10, radius=1.0) == np.inf


def test_asymptotic_bound_is_a_valid_upper_bound_on_exact_tail():
    # exp(-(d-2) g^2/4) should upper-bound the true cap-complement measure
    # 1 - cap_measure(theta0 + phi, d), for reasonably large d / small phi where the
    # asymptotic approximation is expected to hold as a genuine bound.
    d = 50
    theta0 = cap_angle_for_measure(0.5, d)
    for phi_deg in [1, 3, 5, 8]:
        phi = np.radians(phi_deg)
        exact_tail = 1 - cap_measure(theta0 + phi, d)
        eps = chord_from_geodesic(1.0, phi)
        bound = levy_bound_asymptotic(eps, d, radius=1.0)
        assert bound >= exact_tail - 1e-9
