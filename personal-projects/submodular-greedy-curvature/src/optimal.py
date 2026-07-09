"""Exact optimum for cardinality-constrained set-function maximization via
brute-force enumeration. Only tractable for small n; the experiment keeps n
small enough (<= 18) for this to run in well under a second per instance."""

from itertools import combinations


def brute_force_opt(f, k):
    if k < 0:
        raise ValueError("k must be non-negative")
    k = min(k, f.n)
    if k == 0:
        return 0.0, ()

    best_val, best_S = -1.0, None
    for S in combinations(range(f.n), k):
        v = f.value(set(S))
        if v > best_val:
            best_val, best_S = v, S
    return best_val, best_S
