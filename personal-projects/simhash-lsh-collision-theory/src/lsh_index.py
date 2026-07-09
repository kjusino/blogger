"""An AND-OR LSH index built from L independent tables, each keyed by a
length-k random-hyperplane signature. Candidates = union of same-bucket
points across all L tables. This is the classic Indyk-Motwani construction,
not a wrapper around an existing ANN library.
"""
from __future__ import annotations

import numpy as np

from .hyperplane_hash import bits_to_int, hash_bits, random_hyperplanes


def _group_ids_by_key(keys: np.ndarray) -> dict[int, np.ndarray]:
    """Vectorized equivalent of `for i, k in enumerate(keys): table[k].append(i)`,
    built via a single argsort instead of a Python-level loop over rows (which
    dominates runtime once n reaches the tens of thousands)."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, boundaries)
    unique_keys = sorted_keys[np.r_[0, boundaries]]
    return {int(k): g for k, g in zip(unique_keys, groups)}


class LSHIndex:
    def __init__(self, dim: int, k: int, L: int, rng: np.random.Generator):
        self.dim = dim
        self.k = k
        self.L = L
        self.hyperplanes = [random_hyperplanes(k, dim, rng) for _ in range(L)]
        self.tables: list[dict[int, np.ndarray]] = [{} for _ in range(L)]
        self.n_indexed = 0

    def index(self, X: np.ndarray) -> None:
        """Insert all rows of X (n, dim) with ids 0..n-1."""
        n = X.shape[0]
        for table_idx in range(self.L):
            bits = hash_bits(X, self.hyperplanes[table_idx])
            keys = bits_to_int(bits)
            self.tables[table_idx] = _group_ids_by_key(keys)
        self.n_indexed = n

    def query_candidates(self, q: np.ndarray, exclude_id: int | None = None) -> set[int]:
        """Union of bucket-mates of q across all L tables."""
        candidates: set[int] = set()
        for table_idx in range(self.L):
            bits = hash_bits(q[None, :], self.hyperplanes[table_idx])
            key = int(bits_to_int(bits)[0])
            candidates.update(self.tables[table_idx].get(key, []))
        candidates.discard(exclude_id)
        return candidates

    def collides_with(self, q: np.ndarray, target_id: int) -> bool:
        """True iff target_id appears in q's candidate set (i.e. at least one
        of the L tables put them in the same bucket)."""
        for table_idx in range(self.L):
            bits_q = hash_bits(q[None, :], self.hyperplanes[table_idx])
            key_q = int(bits_to_int(bits_q)[0])
            if target_id in self.tables[table_idx].get(key_q, []):
                return True
        return False


def brute_force_nearest(query: np.ndarray, dataset: np.ndarray) -> tuple[int, float]:
    """Exact 1-NN by cosine similarity (ground truth for recall/precision)."""
    q_norm = query / np.linalg.norm(query)
    d_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    sims = d_norm @ q_norm
    best = int(np.argmax(sims))
    return best, float(sims[best])
