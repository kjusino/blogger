"""Closed-form (independence-approximation) predictions used to validate the
simulator, and to compute the exact (non-sampled) expected weighted FPR of a
given k-assignment for a fixed bit array size and insertion budget.
"""
from __future__ import annotations

import math


def load_factor(num_bits: int, total_insertions: int) -> float:
    """P(a given bit is still 0) complement, i.e. fraction of bits set,
    under the standard independent-uniform-hashing approximation."""
    return 1.0 - math.exp(-total_insertions / num_bits)


def item_fpr(load: float, k: int) -> float:
    """False positive probability for an item checked with k hash functions,
    given bit array load factor ``load``."""
    return load ** k


def expected_weighted_fpr(weights: dict[str, float], ks: dict[str, int], load: float) -> float:
    """sum_i w_i * item_fpr(load, k_i), renormalized over the given keys."""
    total_w = sum(weights[key] for key in ks)
    if total_w == 0:
        return 0.0
    return sum(weights[key] * item_fpr(load, k) for key, k in ks.items()) / total_w
