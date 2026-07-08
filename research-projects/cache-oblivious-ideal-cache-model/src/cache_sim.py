"""
An "ideal cache" simulator, per Frigo, Leiserson, Prokop & Ramachandran (1999).

The ideal-cache model abstracts a single level of memory hierarchy as:
  - a cache holding M words, organized into L = M // B fully-associative lines
  - each line holds B consecutive words (a "block")
  - optimal replacement is assumed in the theory; the FLPR99 paper's central
    lemma is that LRU is within a constant factor (2x) of optimal, so we
    simulate LRU directly and treat its miss count as the quantity the
    Theta(...) bounds describe.

Address space is modeled in *words*, not bytes -- this matches how the
cache-oblivious literature states its bounds (Theta(n^3 / (B * sqrt(M)))
counts block transfers, agnostic to word size).
"""

from collections import OrderedDict


class IdealCache:
    """LRU simulation of a fully-associative cache with block size B and
    capacity M words (L = M // B cache lines)."""

    def __init__(self, M, B):
        if B <= 0 or M <= 0:
            raise ValueError("M and B must be positive")
        if M < B:
            raise ValueError("cache capacity M must be at least one block (B)")
        self.M = M
        self.B = B
        self.L = M // B  # number of cache lines
        self._resident = OrderedDict()  # block_id -> None, recency-ordered
        self.hits = 0
        self.misses = 0

    def access(self, address):
        """Touch one word at `address` (a nonnegative int). Returns True on
        hit, False on miss. Updates recency (LRU) and hit/miss counters."""
        block = address // self.B
        if block in self._resident:
            self._resident.move_to_end(block)
            self.hits += 1
            return True
        self.misses += 1
        self._resident[block] = None
        if len(self._resident) > self.L:
            self._resident.popitem(last=False)
        return False

    @property
    def total_accesses(self):
        return self.hits + self.misses

    def reset_counters(self):
        self.hits = 0
        self.misses = 0


class Array1D:
    """A flat word-addressed array occupying `size` words starting at
    `base_address` in a shared address space, instrumented against a
    shared IdealCache on every element read/write."""

    __slots__ = ("cache", "base", "size", "data")

    def __init__(self, cache, base_address, size, init=0.0):
        self.cache = cache
        self.base = base_address
        self.size = size
        self.data = [init] * size

    def get(self, i):
        self.cache.access(self.base + i)
        return self.data[i]

    def set(self, i, value):
        self.cache.access(self.base + i)
        self.data[i] = value


class Matrix:
    """Row-major n x n matrix backed by an instrumented Array1D."""

    __slots__ = ("n", "arr")

    def __init__(self, cache, base_address, n, init=0.0):
        self.n = n
        self.arr = Array1D(cache, base_address, n * n, init=init)

    def get(self, i, j):
        return self.arr.get(i * self.n + j)

    def set(self, i, j, value):
        self.arr.set(i * self.n + j, value)

    def to_list(self):
        n = self.n
        return [[self.arr.data[i * n + j] for j in range(n)] for i in range(n)]

    @classmethod
    def from_list(cls, cache, base_address, rows):
        n = len(rows)
        m = cls(cache, base_address, n)
        for i in range(n):
            for j in range(n):
                m.arr.data[i * n + j] = rows[i][j]
        return m
