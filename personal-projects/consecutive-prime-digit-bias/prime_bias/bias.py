"""Consecutive-prime last-digit bias statistics.

Every prime greater than 5 ends in 1, 3, 7, or 9. If consecutive primes'
last digits behaved like independent draws from {1, 3, 7, 9}, each of the
16 ordered pairs (last digit of p_n, last digit of p_{n+1}) would occur
with frequency 1/16, and in particular the four "repeat" pairs (a, a)
would together account for 4/16 = 1/4 of all consecutive pairs. Lemke
Oliver and Soundararajan (PNAS, 2016) showed primes conspicuously avoid
repeating their last digit far more than this naive model predicts.
"""

from typing import Tuple

import numpy as np
from scipy import stats

DIGITS: Tuple[int, ...] = (1, 3, 7, 9)
NULL_SAME_DIGIT_FRACTION = 0.25


def consecutive_pairs(primes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a sorted prime array into aligned (p_n, p_{n+1}) arrays.

    Primes 2 and 5 are dropped since they don't end in 1/3/7/9, and 3 has
    no defined "last digit" partner category either, so we simply restrict
    to primes > 5.
    """
    filtered = primes[primes > 5]
    if filtered.size < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return filtered[:-1], filtered[1:]


def last_digit_matrix(p_arr: np.ndarray, q_arr: np.ndarray) -> np.ndarray:
    """4x4 count matrix M[i, j] = #pairs with p % 10 == DIGITS[i], q % 10 == DIGITS[j]."""
    matrix = np.zeros((len(DIGITS), len(DIGITS)), dtype=np.int64)
    if p_arr.size == 0:
        return matrix
    pd = p_arr % 10
    qd = q_arr % 10
    for i, a in enumerate(DIGITS):
        p_mask = pd == a
        for j, b in enumerate(DIGITS):
            matrix[i, j] = int(np.sum(p_mask & (qd == b)))
    return matrix


def same_digit_fraction(matrix: np.ndarray) -> Tuple[float, int]:
    """Return (fraction of pairs on the diagonal, total pair count)."""
    total = int(matrix.sum())
    if total == 0:
        return float("nan"), 0
    same = int(np.trace(matrix))
    return same / total, total


def binomial_bias_test(same_count: int, total: int, null_p: float = NULL_SAME_DIGIT_FRACTION) -> float:
    """One-sided binomial test p-value for "same-digit fraction is < null_p"."""
    if total == 0:
        return float("nan")
    result = stats.binomtest(same_count, total, null_p, alternative="less")
    return float(result.pvalue)


def uniform_chisquare_test(matrix: np.ndarray) -> Tuple[float, float]:
    """Chi-square goodness-of-fit of the full 4x4 matrix against uniform 1/16."""
    total = matrix.sum()
    if total == 0:
        return float("nan"), float("nan")
    expected = np.full(matrix.shape, total / (len(DIGITS) ** 2))
    chi2, p = stats.chisquare(matrix.flatten(), expected.flatten())
    return float(chi2), float(p)
