import numpy as np
import pytest

from src.theory import phase_transition_delta, statistical_dimension_fraction


def test_boundary_values():
    assert phase_transition_delta(0.0) == 0.0
    assert phase_transition_delta(1.0) == 1.0


def test_monotone_increasing():
    rhos = np.linspace(0.01, 0.99, 25)
    deltas = [phase_transition_delta(r) for r in rhos]
    assert all(a <= b + 1e-9 for a, b in zip(deltas, deltas[1:]))


def test_bounded_in_unit_interval():
    for rho in np.linspace(0.0, 1.0, 21):
        d = phase_transition_delta(rho)
        assert 0.0 <= d <= 1.0


def test_concave_like_shape_below_diagonal():
    # A rho-sparse vector never needs more measurements than its ambient
    # dimension, and (for rho < 1) strictly fewer than n, i.e. delta < 1.
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert phase_transition_delta(rho) < 1.0
        assert phase_transition_delta(rho) > rho  # always costs more than pure info content... not a proof, sanity only


def test_statistical_dimension_fraction_matches_manual_formula():
    from scipy.stats import norm

    rho, lam = 0.2, 0.5
    expected = rho * (1 + lam**2) + (1 - rho) * 2 * (
        (1 + lam**2) * norm.cdf(-lam) - lam * norm.pdf(lam)
    )
    assert statistical_dimension_fraction(rho, lam) == pytest.approx(expected)


def test_statistical_dimension_fraction_vectorized_over_lam():
    lams = np.array([0.0, 0.5, 1.0, 2.0])
    out = statistical_dimension_fraction(0.3, lams)
    assert out.shape == lams.shape
    for i, lam in enumerate(lams):
        assert out[i] == pytest.approx(statistical_dimension_fraction(0.3, lam))


def test_lam_zero_reduces_to_full_dimension():
    # f(rho, 0) = rho*1 + (1-rho)*2*Phi(0) = rho + (1-rho) = 1, since Phi(0)=0.5
    assert statistical_dimension_fraction(0.4, 0.0) == pytest.approx(1.0)


def test_invalid_rho_raises():
    with pytest.raises(ValueError):
        statistical_dimension_fraction(1.5, 0.0)
    with pytest.raises(ValueError):
        statistical_dimension_fraction(-0.1, 0.0)


def test_invalid_lam_raises():
    with pytest.raises(ValueError):
        statistical_dimension_fraction(0.3, -1.0)


def test_known_reference_points():
    # Sanity-checked against the shape of published Donoho-Tanner /
    # ALMT phase-transition diagrams (rho = k/n convention): the curve
    # rises steeply from the origin and flattens as rho -> 1.
    assert phase_transition_delta(0.1) == pytest.approx(0.3288, abs=5e-3)
    assert phase_transition_delta(0.5) == pytest.approx(0.8313, abs=5e-3)
