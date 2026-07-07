import numpy as np
import pytest

from src.theory import exact_risk, bias_variance_exact, asymptotic_risk


def test_exact_risk_singular_at_and_near_threshold():
    assert exact_risk(n=50, p=50, r2=1.0, sigma2=1.0) == np.inf
    assert exact_risk(n=50, p=49, r2=1.0, sigma2=1.0) == np.inf  # n-p-1==0
    assert exact_risk(n=50, p=51, r2=1.0, sigma2=1.0) == np.inf  # p-n-1==0


def test_exact_risk_matches_manual_formula_below_threshold():
    n, p, sigma2 = 100, 40, 1.5
    expected = sigma2 * p / (n - p - 1)
    assert exact_risk(n, p, r2=1.0, sigma2=sigma2) == pytest.approx(expected)


def test_exact_risk_matches_manual_formula_above_threshold():
    n, p, r2, sigma2 = 100, 250, 2.0, 1.5
    expected = sigma2 * n / (p - n - 1) + r2 * (1 - n / p)
    assert exact_risk(n, p, r2=r2, sigma2=sigma2) == pytest.approx(expected)


def test_bias_variance_decomposition_sums_to_exact_risk():
    for n, p in [(80, 30), (30, 80), (200, 199 - 100)]:
        bias2, variance = bias_variance_exact(n, p, r2=1.3, sigma2=0.7)
        total = exact_risk(n, p, r2=1.3, sigma2=0.7)
        if np.isfinite(total):
            assert bias2 + variance == pytest.approx(total)


def test_bias_is_zero_when_underparameterized():
    bias2, _variance = bias_variance_exact(n=100, p=40, r2=5.0, sigma2=1.0)
    assert bias2 == 0.0


def test_risk_decreases_then_blows_up_then_decreases_again():
    # Qualitative double-descent shape: risk should fall as gamma moves
    # away from 0 towards the underparameterized regime, spike near the
    # threshold, then fall again deep in the overparameterized regime.
    r2, sigma2 = 1.0, 1.0
    r_low = asymptotic_risk(0.2, r2, sigma2)
    r_near_below = asymptotic_risk(0.99, r2, sigma2)
    r_near_above = asymptotic_risk(1.01, r2, sigma2)
    r_high = asymptotic_risk(5.0, r2, sigma2)
    assert r_low < r_near_below
    assert r_near_below > r_high
    assert r_near_above > r_high
    assert r_high < r_near_above and r_high < r_near_below


def test_asymptotic_risk_matches_exact_risk_in_the_large_np_limit():
    # For large but finite n, p at fixed gamma, the finite-sample exact
    # formula should converge to the asymptotic closed form.
    sigma2, r2, gamma = 1.0, 1.0, 3.0
    n = 20000
    p = round(gamma * n)
    exact = exact_risk(n, p, r2, sigma2)
    asymp = asymptotic_risk(gamma, r2, sigma2)
    assert exact == pytest.approx(asymp, rel=1e-3)


def test_asymptotic_risk_undefined_exactly_at_threshold():
    assert asymptotic_risk(1.0, r2=1.0, sigma2=1.0) == np.inf


def test_projection_onto_random_subspace_matches_bias_formula():
    """Independent Monte Carlo check of the bias derivation itself: the
    expected squared component of a fixed unit vector orthogonal to a
    uniformly random n-dimensional subspace of R^p is exactly 1 - n/p.
    This does not call any src code -- it re-derives the building block
    the theory formulas rely on, from first principles.
    """
    rng = np.random.default_rng(42)
    n, p, trials = 15, 40, 400
    beta0 = np.zeros(p)
    beta0[0] = 1.0  # unit vector

    residual_sq = np.empty(trials)
    for t in range(trials):
        X = rng.standard_normal((n, p))  # rows span a random n-dim subspace
        proj = X.T @ np.linalg.solve(X @ X.T, X @ beta0)
        residual_sq[t] = np.sum((beta0 - proj) ** 2)

    empirical = residual_sq.mean()
    theoretical = 1 - n / p
    stderr = residual_sq.std(ddof=1) / np.sqrt(trials)
    assert abs(empirical - theoretical) < 5 * stderr
