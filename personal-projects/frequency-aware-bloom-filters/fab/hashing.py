"""Deterministic, seedable hash family used by both the Bloom filter and the
Count-Min Sketch. Uses BLAKE2b (fast, cryptographically-sound distribution)
rather than Python's built-in ``hash()``, which is randomized per-process and
unsuitable for reproducible experiments.
"""
from __future__ import annotations

import hashlib


def hash_to_int(key: str, salt: int, seed: int = 0) -> int:
    """Return a 64-bit unsigned integer derived from (seed, salt, key).

    ``salt`` selects an independent hash function within a family (e.g. the
    i-th of k hash functions for a Bloom filter, or the i-th of d rows of a
    Count-Min Sketch). ``seed`` selects an independent family altogether
    (e.g. to run multiple randomized trials of the same experiment).
    """
    digest = hashlib.blake2b(
        f"{key}".encode("utf-8"),
        digest_size=8,
        person=f"fab{seed}".encode("utf-8")[:16],
        salt=f"{salt}".encode("utf-8")[:16],
    ).digest()
    return int.from_bytes(digest, "big")


def hash_to_bucket(key: str, salt: int, num_buckets: int, seed: int = 0) -> int:
    return hash_to_int(key, salt, seed) % num_buckets
