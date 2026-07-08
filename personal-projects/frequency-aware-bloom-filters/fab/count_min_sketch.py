"""A standard Count-Min Sketch (Cormode & Muthukrishnan, 2005) used here to
estimate per-key access frequency from a sample of a query/access stream,
without storing a full frequency table.

Guarantees (for width w, depth d, stream length L):
  - estimate(x) >= true_count(x)               always (never underestimates)
  - estimate(x) <= true_count(x) + eps*L        with probability >= 1 - delta
    where eps = e/w and delta = e^{-d} (e = Euler's number)
"""
from __future__ import annotations

import math

import numpy as np

from .hashing import hash_to_bucket


class CountMinSketch:
    def __init__(self, width: int, depth: int, seed: int = 0):
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        self.width = width
        self.depth = depth
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.total_count = 0

    def update(self, item: str, count: int = 1) -> None:
        for row in range(self.depth):
            col = hash_to_bucket(item, salt=1000 + row, num_buckets=self.width, seed=self.seed)
            self.table[row, col] += count
        self.total_count += count

    def estimate(self, item: str) -> int:
        vals = [
            self.table[row, hash_to_bucket(item, salt=1000 + row, num_buckets=self.width, seed=self.seed)]
            for row in range(self.depth)
        ]
        return int(min(vals))

    def error_bound(self) -> tuple[float, float]:
        """(eps, delta): with probability >= 1-delta, estimate <= true + eps*L."""
        eps = math.e / self.width
        delta = math.e ** (-self.depth)
        return eps, delta

    @classmethod
    def from_stream(cls, stream: list[str], width: int, depth: int, seed: int = 0) -> "CountMinSketch":
        cms = cls(width=width, depth=depth, seed=seed)
        for item in stream:
            cms.update(item)
        return cms
