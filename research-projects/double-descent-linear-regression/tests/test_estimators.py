import numpy as np

from src.estimators import fit_min_norm, fit_ridge


def test_fit_min_norm_matches_ols_when_overdetermined():
    rng = np.random.default_rng(0)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    beta0 = rng.standard_normal(p)
    y = X @ beta0 + 0.1 * rng.standard_normal(n)

    beta_hat = fit_min_norm(X, y)
    beta_ols = np.linalg.solve(X.T @ X, X.T @ y)
    assert np.allclose(beta_hat, beta_ols, atol=1e-8)


def test_fit_min_norm_interpolates_when_underdetermined():
    rng = np.random.default_rng(1)
    n, p = 20, 200
    X = rng.standard_normal((n, p))
    beta0 = rng.standard_normal(p)
    y = X @ beta0  # noiseless: an exact interpolator must reproduce y

    beta_hat = fit_min_norm(X, y)
    assert np.allclose(X @ beta_hat, y, atol=1e-6)


def test_fit_min_norm_is_minimum_norm_among_interpolators():
    rng = np.random.default_rng(2)
    n, p = 15, 100
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)

    beta_hat = fit_min_norm(X, y)
    # Any other interpolating solution beta_hat + v with X v = 0 must have
    # at least as large a norm.
    null_direction = rng.standard_normal(p)
    null_direction -= X.T @ np.linalg.solve(X @ X.T, X @ null_direction)
    perturbed = beta_hat + 0.5 * null_direction
    assert np.allclose(X @ perturbed, y, atol=1e-6)
    assert np.linalg.norm(perturbed) >= np.linalg.norm(beta_hat) - 1e-8


def test_fit_ridge_matches_min_norm_at_lambda_zero():
    rng = np.random.default_rng(3)
    n, p = 30, 60
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    assert np.allclose(fit_ridge(X, y, 0.0), fit_min_norm(X, y), atol=1e-8)


def test_fit_ridge_shrinks_coefficient_norm():
    rng = np.random.default_rng(4)
    n, p = 50, 30
    X = rng.standard_normal((n, p))
    beta0 = rng.standard_normal(p)
    y = X @ beta0 + rng.standard_normal(n)

    beta_lo = fit_ridge(X, y, 0.1)
    beta_hi = fit_ridge(X, y, 50.0)
    assert np.linalg.norm(beta_hi) < np.linalg.norm(beta_lo)


def test_fit_ridge_agrees_across_woodbury_and_direct_forms():
    # p > n uses the Woodbury branch; p < n uses the direct normal-equations
    # branch. Cross-check both against a brute-force (X^T X + lam I) solve
    # padded to be square, to make sure the two code paths agree.
    rng = np.random.default_rng(5)
    lam = 2.5
    for n, p in [(10, 40), (40, 10)]:
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        beta_hat = fit_ridge(X, y, lam)
        brute = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
        assert np.allclose(beta_hat, brute, atol=1e-6)
