import numpy as np
import pytest

from src.data import make_dataset


def test_shapes_and_labels():
    rng = np.random.default_rng(0)
    X, y = make_dataset(n=100, d=5, rng=rng)
    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert set(np.unique(y).tolist()) == {0.0, 1.0}


def test_deterministic_given_seed():
    X1, y1 = make_dataset(n=50, d=3, rng=np.random.default_rng(42))
    X2, y2 = make_dataset(n=50, d=3, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    X1, _ = make_dataset(n=50, d=3, rng=np.random.default_rng(1))
    X2, _ = make_dataset(n=50, d=3, rng=np.random.default_rng(2))
    assert not np.allclose(X1, X2)


def test_requires_explicit_rng():
    with pytest.raises(ValueError):
        make_dataset(n=10, d=2, rng=None)


def test_class_balance_roughly_even():
    rng = np.random.default_rng(7)
    _, y = make_dataset(n=101, d=4, rng=rng)
    n0 = int(np.sum(y == 0.0))
    n1 = int(np.sum(y == 1.0))
    assert n0 + n1 == 101
    assert abs(n0 - n1) <= 1
