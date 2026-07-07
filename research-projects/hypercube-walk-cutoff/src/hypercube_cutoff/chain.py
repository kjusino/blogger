"""Exact simulation of the hypercube walk via the Hamming-weight lumping.

The walk on {0,1}^n (pick a uniform coordinate, resample it to a fresh fair
bit) is invariant under permuting coordinates, and its start state (all
zeros) is itself permutation-invariant. Hence the law of the state at every
time t is exchangeable, and -- since the symmetric group acts transitively on
subsets of a fixed size -- conditioning an exchangeable distribution on the
Hamming weight always yields the uniform distribution over subsets of that
weight. So the Hamming weight W_t is an exact lumping of the chain: it is
itself a birth-death Markov chain on {0, ..., n}, and

    TV(Law(X_t), Uniform({0,1}^n)) == TV(Law(W_t), Binomial(n, 1/2))

exactly, for every t -- not just asymptotically. This collapses an
intractable 2^n-dimensional problem into an (n+1)-dimensional one, letting us
compute the *exact* mixing curve (no Monte Carlo) for n in the thousands.
`full_chain_tv_curve` below brute-forces the actual 2^n-state chain for small
n, used only to validate this lumping claim empirically.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def stationary_weight_pmf(n: int) -> np.ndarray:
    """Binomial(n, 1/2) pmf over weights 0..n -- the weight-marginal of Uniform({0,1}^n)."""
    k = np.arange(n + 1)
    return binom.pmf(k, n, 0.5)


def weight_step(dist: np.ndarray, n: int) -> np.ndarray:
    """Apply one step of the lumped birth-death chain to a weight distribution.

    From weight w: -> w-1 w.p. w/(2n), -> w+1 w.p. (n-w)/(2n), stays w.p. 1/2.
    """
    if dist.shape != (n + 1,):
        raise ValueError(f"dist must have shape ({n + 1},), got {dist.shape}")
    k = np.arange(n + 1)
    new_dist = 0.5 * dist
    new_dist[:-1] += dist[1:] * (k[1:]) / (2 * n)
    new_dist[1:] += dist[:-1] * (n - k[:-1]) / (2 * n)
    return new_dist


def tv_curve(n: int, t_values, start_weight: int = 0) -> np.ndarray:
    """Exact TV distance of the lumped chain to stationarity at each t in t_values.

    t_values need not be sorted or unique; returned in the same order.
    """
    t_arr = np.asarray(t_values, dtype=int)
    if np.any(t_arr < 0):
        raise ValueError("t values must be nonnegative")
    if not (0 <= start_weight <= n):
        raise ValueError("start_weight must be in [0, n]")

    order = np.argsort(t_arr)
    sorted_t = t_arr[order]
    pi = stationary_weight_pmf(n)

    dist = np.zeros(n + 1)
    dist[start_weight] = 1.0
    results = np.empty(len(t_arr))

    step = 0
    for pos, target_t in zip(order, sorted_t):
        while step < target_t:
            dist = weight_step(dist, n)
            step += 1
        results[pos] = 0.5 * np.sum(np.abs(dist - pi))
    return results


def full_chain_step(dist: np.ndarray, n: int) -> np.ndarray:
    """One exact step of the full 2^n-state chain (brute force; small n only)."""
    size = 1 << n
    if dist.shape != (size,):
        raise ValueError(f"dist must have shape ({size},), got {dist.shape}")
    all_idx = np.arange(size)
    new_dist = np.zeros(size)
    for i in range(n):
        mask = 1 << i
        is_zero = (all_idx & mask) == 0
        xs0 = all_idx[is_zero]
        xs1 = xs0 | mask
        contribution = (dist[xs0] + dist[xs1]) / (2 * n)
        new_dist[xs0] += contribution
        new_dist[xs1] += contribution
    return new_dist


def full_chain_tv_curve(n: int, t_values, start_state: int = 0):
    """Brute-force exact TV curve on the full 2^n-state chain, plus the weight
    marginal at each requested t (for comparing against the lumped chain)."""
    if n > 14:
        raise ValueError("full_chain_tv_curve is brute force -- keep n <= 14")
    t_arr = np.asarray(t_values, dtype=int)
    order = np.argsort(t_arr)
    sorted_t = t_arr[order]

    size = 1 << n
    dist = np.zeros(size)
    dist[start_state] = 1.0
    uniform = np.full(size, 1.0 / size)

    popcount = np.array([bin(x).count("1") for x in range(size)])

    tv_results = np.empty(len(t_arr))
    weight_marginals = np.empty((len(t_arr), n + 1))

    step = 0
    for pos, target_t in zip(order, sorted_t):
        while step < target_t:
            dist = full_chain_step(dist, n)
            step += 1
        tv_results[pos] = 0.5 * np.sum(np.abs(dist - uniform))
        wm = np.zeros(n + 1)
        np.add.at(wm, popcount, dist)
        weight_marginals[pos] = wm
    return tv_results, weight_marginals
