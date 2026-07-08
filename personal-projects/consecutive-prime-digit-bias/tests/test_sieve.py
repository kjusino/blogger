import numpy as np

from prime_bias.sieve import sieve_primes


def test_small_primes_match_known_list():
    expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert list(sieve_primes(29)) == expected


def test_no_composites_included():
    primes = set(sieve_primes(100))
    composites = {4, 6, 8, 9, 10, 12, 15, 21, 25, 49, 91, 99, 100}
    assert composites.isdisjoint(primes)


def test_prime_counting_function_known_values():
    # pi(x) reference values (OEIS A000720).
    assert len(sieve_primes(10)) == 4
    assert len(sieve_primes(100)) == 25
    assert len(sieve_primes(1000)) == 168
    assert len(sieve_primes(10_000)) == 1229
    assert len(sieve_primes(1_000_000)) == 78498


def test_below_two_is_empty():
    assert sieve_primes(1).size == 0
    assert sieve_primes(0).size == 0


def test_result_is_sorted_and_unique():
    primes = sieve_primes(10_000)
    assert np.all(np.diff(primes) > 0)


def test_perfect_square_upper_bound():
    # n_max itself a perfect square (100 = 10^2) must not misfire the sqrt loop bound.
    primes = sieve_primes(100)
    assert primes[-1] == 97
