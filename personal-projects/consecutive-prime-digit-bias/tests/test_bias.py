import numpy as np
import pytest

from prime_bias.bias import (
    DIGITS,
    binomial_bias_test,
    consecutive_pairs,
    last_digit_matrix,
    same_digit_fraction,
    uniform_chisquare_test,
)


def test_consecutive_pairs_drops_2_3_5():
    primes = np.array([2, 3, 5, 7, 11, 13, 17])
    p_arr, q_arr = consecutive_pairs(primes)
    # only primes > 5 survive: 7, 11, 13, 17 -> 3 consecutive pairs
    assert list(p_arr) == [7, 11, 13]
    assert list(q_arr) == [11, 13, 17]


def test_consecutive_pairs_too_short():
    p_arr, q_arr = consecutive_pairs(np.array([2, 3, 5]))
    assert p_arr.size == 0
    assert q_arr.size == 0


def test_last_digit_matrix_known_pairs():
    # 11 -> 13 (1,3), 13 -> 17 (3,7), 17 -> 19 (7,9), 19 -> 23 (9,3)
    p_arr = np.array([11, 13, 17, 19])
    q_arr = np.array([13, 17, 19, 23])
    matrix = last_digit_matrix(p_arr, q_arr)
    assert matrix.sum() == 4
    idx = {d: i for i, d in enumerate(DIGITS)}
    assert matrix[idx[1], idx[3]] == 1
    assert matrix[idx[3], idx[7]] == 1
    assert matrix[idx[7], idx[9]] == 1
    assert matrix[idx[9], idx[3]] == 1


def test_same_digit_fraction_all_same():
    matrix = np.zeros((4, 4), dtype=np.int64)
    matrix[0, 0] = 10  # all (1,1) pairs
    frac, total = same_digit_fraction(matrix)
    assert total == 10
    assert frac == pytest.approx(1.0)


def test_same_digit_fraction_empty():
    matrix = np.zeros((4, 4), dtype=np.int64)
    frac, total = same_digit_fraction(matrix)
    assert total == 0
    assert np.isnan(frac)


def test_uniform_matrix_has_high_chisquare_pvalue():
    matrix = np.full((4, 4), 100, dtype=np.int64)
    chi2, p = uniform_chisquare_test(matrix)
    assert chi2 == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_skewed_matrix_has_low_chisquare_pvalue():
    matrix = np.full((4, 4), 100, dtype=np.int64)
    matrix[0, 0] = 10_000  # heavily skewed toward (1,1)
    chi2, p = uniform_chisquare_test(matrix)
    assert p < 1e-6


def test_binomial_bias_test_detects_significant_underrepresentation():
    # 2000 pairs, only 300 "same digit" -> fraction 0.15, well below null 0.25
    p_value = binomial_bias_test(same_count=300, total=2000)
    assert p_value < 1e-6


def test_binomial_bias_test_null_when_at_expected_rate():
    p_value = binomial_bias_test(same_count=25, total=100)  # exactly 0.25
    assert p_value > 0.4
