"""Vectorized Sieve of Eratosthenes."""

import math

import numpy as np


def sieve_primes(n_max: int) -> np.ndarray:
    """Return a sorted array of all primes <= n_max.

    Uses numpy boolean-array slicing so each prime's multiples are struck
    out in a single vectorized assignment rather than a Python loop over
    multiples, which is what makes sieving up to 10**9 tractable in
    seconds rather than minutes.
    """
    if n_max < 2:
        return np.array([], dtype=np.int64)

    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0:2] = False
    limit = math.isqrt(n_max)
    for p in range(2, limit + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return np.flatnonzero(is_prime).astype(np.int64)
