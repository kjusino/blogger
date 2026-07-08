import numpy as np

from src.subspace import (
    coherent_basis,
    incoherent_basis,
    leverage_scores,
    random_least_squares_system,
)


def test_incoherent_basis_orthonormal():
    rng = np.random.default_rng(0)
    Q = incoherent_basis(200, 10, rng)
    np.testing.assert_allclose(Q.T @ Q, np.eye(10), atol=1e-9)


def test_coherent_basis_orthonormal():
    Q = coherent_basis(200, 10, np.random.default_rng(0))
    np.testing.assert_allclose(Q.T @ Q, np.eye(10), atol=1e-12)


def test_leverage_scores_sum_to_rank():
    rng = np.random.default_rng(1)
    for Q in [incoherent_basis(150, 12, rng), coherent_basis(150, 12, rng)]:
        assert np.isclose(leverage_scores(Q).sum(), 12, atol=1e-9)


def test_coherent_basis_has_maximal_spike():
    n, d = 100, 6
    Q = coherent_basis(n, d, np.random.default_rng(0))
    lev = leverage_scores(Q)
    np.testing.assert_allclose(lev[:d], np.ones(d))
    np.testing.assert_allclose(lev[d:], np.zeros(n - d))


def test_incoherent_basis_leverage_roughly_uniform():
    n, d = 2000, 10
    rng = np.random.default_rng(2)
    Q = incoherent_basis(n, d, rng)
    lev = leverage_scores(Q)
    # No single row should dominate the way it does in the coherent (spiky) case.
    assert lev.max() < 20 * (d / n)


def test_random_least_squares_system_shapes_and_rank():
    n, d = 300, 8
    rng = np.random.default_rng(3)
    Q = incoherent_basis(n, d, rng)
    A, b, x_true = random_least_squares_system(Q, rng)
    assert A.shape == (n, d)
    assert b.shape == (n,)
    assert x_true.shape == (d,)
    assert np.linalg.matrix_rank(A) == d
