import numpy as np
import pytest

from bbp_transition.theory import (
    bbp_threshold,
    mp_edge,
    theoretical_top_eigenvalue,
    theoretical_alignment_sq,
)


def test_bbp_threshold_basic():
    assert bbp_threshold(0.25) == pytest.approx(0.5)
    assert bbp_threshold(1.0) == pytest.approx(1.0)


def test_bbp_threshold_rejects_nonpositive_c():
    with pytest.raises(ValueError):
        bbp_threshold(0.0)
    with pytest.raises(ValueError):
        bbp_threshold(-1.0)


def test_mp_edge_matches_known_values():
    assert mp_edge(0.25) == pytest.approx((1 + 0.5) ** 2)
    assert mp_edge(1.0) == pytest.approx(4.0)


def test_eigenvalue_below_threshold_equals_mp_edge():
    c = 0.3
    thr = bbp_threshold(c)
    lam_below = np.array([0.0, thr * 0.5, thr * 0.99, thr])
    result = theoretical_top_eigenvalue(lam_below, c)
    np.testing.assert_allclose(result, mp_edge(c))


def test_alignment_below_threshold_is_zero():
    c = 0.3
    thr = bbp_threshold(c)
    lam_below = np.array([0.0, thr * 0.5, thr * 0.99, thr])
    result = theoretical_alignment_sq(lam_below, c)
    np.testing.assert_allclose(result, 0.0)


def test_continuity_at_threshold():
    """Eigenvalue and alignment formulas must agree from both sides at lam=lam*
    -- this is the standard sanity check that the two-regime formula is
    self-consistent (Baik & Silverstein 2006 / Paul 2007)."""
    for c in [0.1, 0.3, 0.5, 0.8]:
        thr = bbp_threshold(c)
        eps = 1e-6
        eig_below = theoretical_top_eigenvalue(thr - eps, c)
        eig_at = theoretical_top_eigenvalue(thr, c)
        eig_above = theoretical_top_eigenvalue(thr + eps, c)
        assert eig_below == pytest.approx(eig_at, abs=1e-4)
        assert eig_above == pytest.approx(eig_at, abs=1e-4)

        align_at = theoretical_alignment_sq(thr, c)
        align_above = theoretical_alignment_sq(thr + eps, c)
        assert align_at == pytest.approx(0.0, abs=1e-9)
        assert align_above == pytest.approx(0.0, abs=1e-3)


def test_eigenvalue_supercritical_matches_closed_form():
    c = 0.25
    lam = 2.0
    expected = (1 + lam) * (1 + c / lam)
    result = float(theoretical_top_eigenvalue(lam, c))
    assert result == pytest.approx(expected)


def test_alignment_supercritical_matches_closed_form():
    c = 0.25
    lam = 2.0
    expected = (1 - c / lam**2) / (1 + c / lam)
    result = float(theoretical_alignment_sq(lam, c))
    assert result == pytest.approx(expected)


def test_alignment_is_monotone_increasing_above_threshold():
    c = 0.4
    thr = bbp_threshold(c)
    lams = np.linspace(thr, thr * 10, 50)
    aligns = theoretical_alignment_sq(lams, c)
    assert np.all(np.diff(aligns) >= -1e-12)


def test_alignment_approaches_one_for_large_lambda():
    c = 0.2
    result = theoretical_alignment_sq(1e6, c)
    assert result == pytest.approx(1.0, abs=1e-4)


def test_alignment_always_in_unit_interval():
    c = 0.6
    lams = np.linspace(0, 50, 200)
    aligns = theoretical_alignment_sq(lams, c)
    assert np.all(aligns >= 0.0) and np.all(aligns <= 1.0)


def test_negative_lambda_rejected():
    with pytest.raises(ValueError):
        theoretical_top_eigenvalue(-1.0, 0.3)
    with pytest.raises(ValueError):
        theoretical_alignment_sq(-1.0, 0.3)
