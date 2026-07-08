import numpy as np
import pytest

from src.sensing import gaussian_sensing_matrix, sparse_signal


def test_gaussian_sensing_matrix_shape():
    rng = np.random.default_rng(1)
    A = gaussian_sensing_matrix(10, 30, rng)
    assert A.shape == (10, 30)


def test_gaussian_sensing_matrix_scaling():
    # Columns should have expected squared norm ~= 1 (entries ~ N(0, 1/m)).
    rng = np.random.default_rng(2)
    m, n = 500, 20
    A = gaussian_sensing_matrix(m, n, rng)
    col_norms_sq = np.sum(A**2, axis=0)
    assert np.allclose(col_norms_sq, 1.0, atol=0.25)


def test_gaussian_sensing_matrix_rejects_nonpositive_dims():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        gaussian_sensing_matrix(0, 5, rng)
    with pytest.raises(ValueError):
        gaussian_sensing_matrix(5, 0, rng)


def test_sparse_signal_has_exact_sparsity():
    rng = np.random.default_rng(4)
    x = sparse_signal(50, 7, rng)
    assert x.shape == (50,)
    assert np.count_nonzero(x) == 7


def test_sparse_signal_zero_sparsity():
    rng = np.random.default_rng(5)
    x = sparse_signal(20, 0, rng)
    assert np.count_nonzero(x) == 0


def test_sparse_signal_full_sparsity():
    rng = np.random.default_rng(6)
    x = sparse_signal(10, 10, rng)
    assert np.count_nonzero(x) == 10


def test_sparse_signal_rejects_k_greater_than_n():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        sparse_signal(5, 6, rng)


def test_sparse_signal_support_is_random_across_calls():
    rng = np.random.default_rng(8)
    supports = set()
    for _ in range(10):
        x = sparse_signal(100, 3, rng)
        supports.add(tuple(np.nonzero(x)[0]))
    assert len(supports) > 1
