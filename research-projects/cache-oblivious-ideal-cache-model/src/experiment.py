"""
Experiment sweeps for the ideal-cache matmul study.

Four sweeps, each holding two of {n, B, M} fixed and varying the third,
run against all three algorithms (naive / blocked / oblivious):

  E1 "scaling_n": fixed B, M (tall cache); n varies.
      Tests: all three algorithms have miss counts ~ n^3 (same exponent),
      but with very different leading constants.

  E2 "scaling_B": fixed n, M (tall cache for all B in range); B varies.
      Tests: blocked/oblivious misses ~ B^-1; naive is ~flat in B.

  E3 "scaling_M": fixed n, B (tall cache throughout, matrices never fully
      cache-resident); M varies.
      Tests: blocked/oblivious misses ~ M^-0.5; naive is ~flat in M.

  E4 "tall_cache_boundary": fixed n, B; M swept across the B^2 tall-cache
      threshold. Tests that oblivious tracks (freshly-retuned) blocked
      while M >= B^2, and that both algorithms' advantage over naive
      shrinks as the tall-cache assumption is violated (M < B^2).
"""

import random
import time

from .cache_sim import IdealCache, Matrix
from .matmul import ALGORITHMS, default_tile


def _fill_random(matrix, rng):
    for idx in range(len(matrix.arr.data)):
        matrix.arr.data[idx] = rng.uniform(-1.0, 1.0)


def run_single(n, B, M, algorithm, base_case=8, seed=0):
    """Run one (n, B, M, algorithm) configuration and return a result dict.
    Matrix contents are randomized (seeded) but multiplication correctness
    is not re-verified here for speed -- see tests/test_matmul.py."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}; choices: {sorted(ALGORITHMS)}")
    rng = random.Random(seed)
    cache = IdealCache(M=M, B=B)
    A = Matrix(cache, base_address=0, n=n)
    Bm = Matrix(cache, base_address=n * n, n=n)
    C = Matrix(cache, base_address=2 * n * n, n=n)
    _fill_random(A, rng)
    _fill_random(Bm, rng)
    cache.reset_counters()

    tile = default_tile(M, n=n)
    start = time.perf_counter()
    ALGORITHMS[algorithm](A, Bm, C, tile, base_case)
    elapsed = time.perf_counter() - start

    return {
        "algorithm": algorithm,
        "n": n,
        "B": B,
        "M": M,
        "tile": tile,
        "misses": cache.misses,
        "hits": cache.hits,
        "total_accesses": cache.total_accesses,
        "seconds": elapsed,
    }


def run_sweep_n(n_values, B, M, algorithms, seed=0):
    records = []
    for algo in algorithms:
        for n in n_values:
            r = run_single(n, B, M, algo, seed=seed)
            r["experiment"] = "scaling_n"
            records.append(r)
    return records


def run_sweep_B(n, B_values, M, algorithms, seed=0):
    records = []
    for algo in algorithms:
        for B in B_values:
            r = run_single(n, B, M, algo, seed=seed)
            r["experiment"] = "scaling_B"
            records.append(r)
    return records


def run_sweep_M(n, B, M_values, algorithms, seed=0):
    records = []
    for algo in algorithms:
        for M in M_values:
            r = run_single(n, B, M, algo, seed=seed)
            r["experiment"] = "scaling_M"
            records.append(r)
    return records


def run_sweep_tall_cache_boundary(n, B, M_values, algorithms, seed=0):
    records = []
    for algo in algorithms:
        for M in M_values:
            r = run_single(n, B, M, algo, seed=seed)
            r["experiment"] = "tall_cache_boundary"
            r["tall_cache_ratio"] = M / (B ** 2)
            records.append(r)
    return records


def run_sweep_naive_capacity_cliff(n, B, M_values, seed=0):
    """Naive matmul (any single fixed loop order) has automatic spatial
    reuse -- Theta(n^3 / B) misses -- *provided* the cache can hold the
    n cache lines needed to retain one full row-block group across the
    middle loop (L = M / B >= n, i.e. M >= n * B). Below that capacity it
    thrashes to Theta(n^3). This is a step function in M at the threshold
    M = n * B, not a power law, so it gets its own sweep/plot rather than
    a log-log regression."""
    records = []
    for M in M_values:
        r = run_single(n, B, M, "naive", seed=seed)
        r["experiment"] = "naive_capacity_cliff"
        r["capacity_ratio"] = M / (n * B)
        records.append(r)
    return records
