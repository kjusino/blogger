import numpy as np
import pytest

from src.functions import (
    MajorityFunction,
    ParityFunction,
    RandomDNFFunction,
    TribesFunction,
)


def test_parity_matches_product():
    f = ParityFunction(n=5)
    rng = np.random.default_rng(0)
    x = rng.choice([-1, 1], size=(200, 5))
    expected = np.prod(x, axis=1)
    assert np.array_equal(f.evaluate_batch(x), expected)


def test_majority_requires_odd_n():
    with pytest.raises(ValueError):
        MajorityFunction(n=4)


def test_majority_matches_sign_of_sum():
    f = MajorityFunction(n=7)
    rng = np.random.default_rng(1)
    x = rng.choice([-1, 1], size=(300, 7))
    # No ties possible with odd n, so sum is never 0.
    expected = np.where(np.sum(x, axis=1) > 0, 1, -1)
    assert np.array_equal(f.evaluate_batch(x), expected)


def test_tribes_all_true_is_true():
    f = TribesFunction(w=3, s=4)
    x = np.ones((1, 12), dtype=np.int64)
    assert f.evaluate(x[0]) == 1


def test_tribes_all_false_is_false():
    f = TribesFunction(w=3, s=4)
    x = -np.ones((1, 12), dtype=np.int64)
    assert f.evaluate(x[0]) == -1


def test_tribes_one_full_tribe_is_true():
    # w=2, s=3: tribe 0 = [x0,x1], tribe1=[x2,x3], tribe2=[x4,x5]
    f = TribesFunction(w=2, s=3)
    x = np.array([[1, 1, -1, -1, -1, -1]], dtype=np.int64)
    assert f.evaluate(x[0]) == 1


def test_tribes_no_full_tribe_is_false():
    f = TribesFunction(w=2, s=3)
    x = np.array([[1, -1, 1, -1, 1, -1]], dtype=np.int64)
    assert f.evaluate(x[0]) == -1


def test_tribes_n_equals_w_times_s():
    f = TribesFunction(w=5, s=9)
    assert f.n == 45


def test_random_dnf_reproducible_with_seed():
    f1 = RandomDNFFunction(n=10, k=2, m=3, seed=123)
    f2 = RandomDNFFunction(n=10, k=2, m=3, seed=123)
    assert np.array_equal(f1.term_vars, f2.term_vars)
    assert np.array_equal(f1.term_signs, f2.term_signs)


def test_random_dnf_different_seeds_usually_differ():
    f1 = RandomDNFFunction(n=20, k=3, m=5, seed=1)
    f2 = RandomDNFFunction(n=20, k=3, m=5, seed=2)
    assert not np.array_equal(f1.term_vars, f2.term_vars)


def test_random_dnf_output_shape_and_range():
    f = RandomDNFFunction(n=15, k=3, m=4, seed=5)
    rng = np.random.default_rng(2)
    x = rng.choice([-1, 1], size=(50, 15))
    out = f.evaluate_batch(x)
    assert out.shape == (50,)
    assert set(np.unique(out)).issubset({-1, 1})


def test_random_dnf_single_satisfied_term_makes_formula_true():
    f = RandomDNFFunction(n=6, k=2, m=1, seed=9)
    var_i, var_j = f.term_vars[0]
    sign_i, sign_j = f.term_signs[0]
    x = np.full((1, 6), -2, dtype=np.int64)  # placeholder, filled below
    x = np.random.default_rng(3).choice([-1, 1], size=(1, 6))
    x[0, var_i] = sign_i
    x[0, var_j] = sign_j
    assert f.evaluate(x[0]) == 1
