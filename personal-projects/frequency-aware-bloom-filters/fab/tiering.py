"""Two-tier hash-function allocation for a Frequency-Aware Bloom Filter.

Given a per-key "importance" score (either the true Zipf weight, an oracle,
or a Count-Min Sketch estimate), the top ``hot_fraction`` of keys by score
get ``k_hot`` hash functions each; the rest get a ``k_cold`` derived so the
*average* number of hash functions across all keys equals ``k_base`` -- the
same total hashing budget (and therefore the same total memory traffic /
bit-array occupancy) as a uniform baseline Bloom filter using k_base for
every key. This makes every comparison in the experiments an apples-to-
apples one at equal memory.

Proposition (why this helps under skewed query traffic).
Under the standard independence approximation, a Bloom filter with bit
array size m and T total (item, hash) insertions gives item i a false
positive probability f(k_i) = L^{k_i}, where L = 1 - e^{-T/m} < 1 is the
bit load factor. f is convex and strictly decreasing in k. Minimizing the
query-weighted expected FPR  sum_i w_i * f(k_i)  subject to sum_i k_i = T
fixed is a convex resource-allocation problem; by the rearrangement /
Lagrangian argument, the optimum pairs larger k_i with larger w_i (the
marginal reduction in f from one extra hash function is largest where f is
still large, i.e. at small k, and that marginal reduction is worth the most
when w_i is large). A uniform allocation (k_i constant) is optimal only in
the degenerate case where all w_i are equal. Whenever query traffic is
skewed (w_i unequal, e.g. Zipfian), shifting hash-function budget from
low-weight to high-weight keys strictly lowers the weighted expected FPR.
The two-tier scheme here is a simple, discrete instance of that shift.
"""
from __future__ import annotations

import numpy as np


def hot_key_set(scores: dict[str, float], hot_fraction: float) -> set[str]:
    if not 0.0 <= hot_fraction <= 1.0:
        raise ValueError("hot_fraction must be in [0, 1]")
    n = len(scores)
    n_hot = round(hot_fraction * n)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {key for key, _ in ranked[:n_hot]}


def solve_k_cold(hot_fraction: float, k_hot: float, k_base: float) -> float:
    """Solve hot_fraction*k_hot + (1-hot_fraction)*k_cold = k_base for k_cold,
    clamped to >= 1 (every key must get at least one hash function)."""
    if hot_fraction >= 1.0:
        return 1.0
    k_cold = (k_base - hot_fraction * k_hot) / (1.0 - hot_fraction)
    return max(k_cold, 1.0)


def assign_k(
    keys: list[str],
    scores: dict[str, float],
    hot_fraction: float,
    k_hot: int,
    k_base: float,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Return an integer k_i for every key in ``keys``.

    k_cold may be fractional (to hit the exact k_base budget on average); we
    realize that average with per-key randomized rounding (round up with
    probability equal to the fractional part), which is unbiased in
    expectation and converges to the target average as len(keys) grows.
    """
    hot = hot_key_set(scores, hot_fraction)
    k_cold = solve_k_cold(hot_fraction, k_hot, k_base)
    k_cold_floor = int(np.floor(k_cold))
    frac = k_cold - k_cold_floor

    assignment: dict[str, int] = {}
    for key in keys:
        if key in hot:
            assignment[key] = int(k_hot)
        else:
            bump = 1 if rng.random() < frac else 0
            assignment[key] = max(k_cold_floor + bump, 1)
    return assignment


def uniform_k(keys: list[str], k_base: int) -> dict[str, int]:
    return {key: int(k_base) for key in keys}
