import numpy as np
import pytest

from src.patterns import (
    arcsin_law,
    generate_correlated_patterns,
    empirical_pairwise_correlation,
)


def test_pattern_shape_and_alphabet():
    rng = np.random.default_rng(0)
    patterns = generate_correlated_patterns(n=50, p=20, rho=0.3, rng=rng)
    assert patterns.shape == (20, 50)
    assert set(np.unique(patterns).tolist()) <= {-1.0, 1.0}


def test_rho_zero_gives_near_zero_correlation():
    rng = np.random.default_rng(1)
    patterns = generate_correlated_patterns(n=2000, p=200, rho=0.0, rng=rng)
    corr = empirical_pairwise_correlation(patterns)
    assert abs(corr) < 0.02  # i.i.d. patterns: correlation should be ~0


def test_invalid_rho_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_correlated_patterns(n=10, p=5, rho=1.0, rng=rng)
    with pytest.raises(ValueError):
        generate_correlated_patterns(n=10, p=5, rho=-0.1, rng=rng)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
def test_arcsin_law_matches_empirical_correlation(rho):
    """corr(sign(X), sign(Y)) = (2/pi) * arcsin(rho) for jointly Gaussian
    X, Y with correlation rho. Verify numerically, not just by assumption."""
    rng = np.random.default_rng(42)
    n, p, n_trials = 3000, 150, 8
    empirical = []
    for _ in range(n_trials):
        patterns = generate_correlated_patterns(n, p, rho, rng)
        empirical.append(empirical_pairwise_correlation(patterns))
    mean_emp = float(np.mean(empirical))
    theoretical = arcsin_law(rho)
    assert abs(mean_emp - theoretical) < 0.02, (
        f"rho={rho}: empirical={mean_emp:.4f} theoretical={theoretical:.4f}"
    )


def test_arcsin_law_monotonic_and_bounds():
    rhos = np.linspace(0, 0.99, 50)
    vals = arcsin_law(rhos)
    assert np.all(np.diff(vals) >= 0)
    assert vals[0] == pytest.approx(0.0, abs=1e-9)
    assert arcsin_law(1.0) == pytest.approx(1.0, abs=1e-9)
