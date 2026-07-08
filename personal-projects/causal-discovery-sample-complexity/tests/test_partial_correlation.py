import numpy as np
import pytest

from src.partial_correlation import partial_correlation, fisher_z_test, conditional_independence_test


def test_partial_correlation_matches_marginal_correlation_with_empty_cond_set():
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    r = partial_correlation(cov, 0, 1, ())
    assert np.isclose(r, 0.6)


def test_partial_correlation_zero_when_conditionally_independent():
    # Chain X0 -> X1 -> X2 with unit weights and unit noise variance:
    # X0 and X2 are independent given X1.
    b01, b12 = 0.8, 0.7
    var0 = 1.0
    var1 = b01 ** 2 * var0 + 1.0
    cov01 = b01 * var0
    cov12 = b12 * var1
    cov02 = b01 * b12 * var0
    var2 = b12 ** 2 * var1 + 1.0
    cov = np.array([
        [var0, cov01, cov02],
        [cov01, var1, cov12],
        [cov02, cov12, var2],
    ])
    r = partial_correlation(cov, 0, 2, (1,))
    assert abs(r) < 1e-8


def test_partial_correlation_bounded():
    rng = np.random.default_rng(0)
    for _ in range(50):
        A = rng.normal(size=(5, 5))
        cov = A @ A.T + np.eye(5) * 0.1
        r = partial_correlation(cov, 0, 1, (2, 3))
        assert -1.0 <= r <= 1.0


def test_fisher_z_test_low_dof_defaults_to_dependent():
    # n too small relative to conditioning set size: cannot test, must not
    # claim independence.
    assert fisher_z_test(r=0.01, n=5, cond_set_size=4, alpha=0.05) is False


def test_fisher_z_test_detects_independence_for_true_zero_correlation():
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    r = np.corrcoef(x, y)[0, 1]
    assert bool(fisher_z_test(r, n=n, cond_set_size=0, alpha=0.01)) is True


def test_fisher_z_test_rejects_independence_for_strong_correlation():
    assert bool(fisher_z_test(r=0.8, n=200, cond_set_size=0, alpha=0.01)) is False


def test_conditional_independence_test_consistent_with_partial_correlation_and_fisher_z():
    cov = np.array([[1.0, 0.01], [0.01, 1.0]])
    result = conditional_independence_test(cov, 0, 1, (), n=1000, alpha=0.01)
    r = partial_correlation(cov, 0, 1, ())
    expected = fisher_z_test(r, 1000, 0, 0.01)
    assert result == expected
