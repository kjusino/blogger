"""A Bloom filter that supports a *different* number of hash functions per
item (variable k_i), as required for weighted / frequency-aware allocation.

Correctness invariant: whatever k is used to *insert* an item must be used
again to *query* it, or the filter can produce a false negative (which a
standard Bloom filter must never do). Callers are responsible for deriving
k deterministically from the item's identity (see ``fab.tiering``) so the
same k is recovered at query time without storing per-item metadata.
"""
from __future__ import annotations

import numpy as np

from .hashing import hash_to_bucket


class BloomFilter:
    def __init__(self, num_bits: int, seed: int = 0):
        if num_bits <= 0:
            raise ValueError("num_bits must be positive")
        self.num_bits = num_bits
        self.seed = seed
        self.bits = np.zeros(num_bits, dtype=bool)
        self.total_insertions = 0  # sum of k_i over all inserted items (= T)

    def _positions(self, item: str, k: int) -> list[int]:
        if k <= 0:
            raise ValueError("k must be positive")
        return [hash_to_bucket(item, salt, self.num_bits, self.seed) for salt in range(k)]

    def insert(self, item: str, k: int) -> None:
        for pos in self._positions(item, k):
            self.bits[pos] = True
        self.total_insertions += k

    def query(self, item: str, k: int) -> bool:
        return all(self.bits[pos] for pos in self._positions(item, k))

    def load_factor(self) -> float:
        """Fraction of bits currently set to 1."""
        return float(self.bits.mean())
