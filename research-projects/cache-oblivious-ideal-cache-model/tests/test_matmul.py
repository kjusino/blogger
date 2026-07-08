import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.cache_sim import IdealCache, Matrix
from src.matmul import (
    default_tile,
    multiply_blocked,
    multiply_naive,
    multiply_oblivious,
)


def _reference(a_rows, b_rows, n):
    c = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += a_rows[i][k] * b_rows[k][j]
            c[i][j] = s
    return c


def _random_matrix_pair(n, seed):
    rng = random.Random(seed)
    a = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
    b = [[rng.uniform(-3, 3) for _ in range(n)] for _ in range(n)]
    return a, b


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 16, 17])
def test_naive_matches_reference(n):
    a_rows, b_rows = _random_matrix_pair(n, seed=n)
    ref = _reference(a_rows, b_rows, n)
    cache = IdealCache(M=4096, B=8)
    A = Matrix.from_list(cache, 0, a_rows)
    B = Matrix.from_list(cache, n * n, b_rows)
    C = Matrix(cache, 2 * n * n, n)
    multiply_naive(A, B, C)
    got = C.to_list()
    for i in range(n):
        for j in range(n):
            assert got[i][j] == pytest.approx(ref[i][j])


@pytest.mark.parametrize("n,tile", [(1, 1), (5, 2), (8, 3), (16, 5), (17, 4)])
def test_blocked_matches_reference_for_ragged_tiles(n, tile):
    a_rows, b_rows = _random_matrix_pair(n, seed=100 + n)
    ref = _reference(a_rows, b_rows, n)
    cache = IdealCache(M=4096, B=8)
    A = Matrix.from_list(cache, 0, a_rows)
    B = Matrix.from_list(cache, n * n, b_rows)
    C = Matrix(cache, 2 * n * n, n)
    multiply_blocked(A, B, C, tile)
    got = C.to_list()
    for i in range(n):
        for j in range(n):
            assert got[i][j] == pytest.approx(ref[i][j])


@pytest.mark.parametrize("n,base_case", [(1, 1), (5, 1), (8, 3), (16, 4), (17, 5), (23, 8)])
def test_oblivious_matches_reference_for_nonsquare_splits(n, base_case):
    a_rows, b_rows = _random_matrix_pair(n, seed=200 + n)
    ref = _reference(a_rows, b_rows, n)
    cache = IdealCache(M=4096, B=8)
    A = Matrix.from_list(cache, 0, a_rows)
    B = Matrix.from_list(cache, n * n, b_rows)
    C = Matrix(cache, 2 * n * n, n)
    multiply_oblivious(A, B, C, base_case=base_case)
    got = C.to_list()
    for i in range(n):
        for j in range(n):
            assert got[i][j] == pytest.approx(ref[i][j])


def test_all_three_algorithms_agree_with_each_other():
    n = 20
    a_rows, b_rows = _random_matrix_pair(n, seed=7)
    results = {}
    for name, fn, kwargs in [
        ("naive", multiply_naive, {}),
        ("blocked", multiply_blocked, {"tile": 6}),
        ("oblivious", multiply_oblivious, {"base_case": 5}),
    ]:
        cache = IdealCache(M=4096, B=8)
        A = Matrix.from_list(cache, 0, a_rows)
        B = Matrix.from_list(cache, n * n, b_rows)
        C = Matrix(cache, 2 * n * n, n)
        fn(A, B, C, **kwargs)
        results[name] = C.to_list()

    for i in range(n):
        for j in range(n):
            assert results["naive"][i][j] == pytest.approx(results["blocked"][i][j])
            assert results["naive"][i][j] == pytest.approx(results["oblivious"][i][j])


def test_default_tile_positive_and_bounded():
    t = default_tile(M=300)
    assert t >= 1
    assert t <= int((300 / 3) ** 0.5) + 1


def test_default_tile_snaps_to_divisor_of_n():
    t = default_tile(M=4096, n=100)
    assert 100 % t == 0


def test_default_tile_never_zero_even_for_prime_n():
    t = default_tile(M=4096, n=97)  # prime: only divisors are 1 and 97
    assert t >= 1
    assert 97 % t == 0


def test_blocked_matmul_uses_fewer_or_equal_misses_than_naive_at_scale():
    # a coarse sanity check that blocking actually helps once the problem
    # is bigger than the cache -- the core motivating claim of the study.
    n = 64
    a_rows, b_rows = _random_matrix_pair(n, seed=1)
    B_size = 8
    M = 512  # << n^2, so blocking's temporal reuse should matter
    naive_cache = IdealCache(M=M, B=B_size)
    A = Matrix.from_list(naive_cache, 0, a_rows)
    Bm = Matrix.from_list(naive_cache, n * n, b_rows)
    C = Matrix(naive_cache, 2 * n * n, n)
    multiply_naive(A, Bm, C)

    blocked_cache = IdealCache(M=M, B=B_size)
    A2 = Matrix.from_list(blocked_cache, 0, a_rows)
    B2 = Matrix.from_list(blocked_cache, n * n, b_rows)
    C2 = Matrix(blocked_cache, 2 * n * n, n)
    multiply_blocked(A2, B2, C2, default_tile(M, n=n))

    assert blocked_cache.misses < naive_cache.misses
