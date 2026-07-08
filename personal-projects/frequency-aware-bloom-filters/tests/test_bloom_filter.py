import math

import numpy as np
import pytest

from fab.bloom_filter import BloomFilter
from fab.theory import item_fpr, load_factor


def test_no_false_negatives():
    """A Bloom filter must never say 'not present' for an item it inserted,
    as long as query uses the same k that was used at insert time."""
    bf = BloomFilter(num_bits=2000, seed=1)
    items = [f"item-{i}" for i in range(200)]
    for i, item in enumerate(items):
        k = 3 + (i % 5)  # vary k per item, as FAB-filters do
        bf.insert(item, k)

    for i, item in enumerate(items):
        k = 3 + (i % 5)
        assert bf.query(item, k) is True


def test_hash_positions_deterministic():
    bf = BloomFilter(num_bits=1000, seed=42)
    pos1 = bf._positions("hello", k=5)
    pos2 = bf._positions("hello", k=5)
    assert pos1 == pos2


def test_different_seeds_give_different_filters():
    bf1 = BloomFilter(num_bits=1000, seed=1)
    bf2 = BloomFilter(num_bits=1000, seed=2)
    assert bf1._positions("hello", k=5) != bf2._positions("hello", k=5)


def test_total_insertions_tracks_sum_of_k():
    bf = BloomFilter(num_bits=5000, seed=0)
    ks = [3, 5, 7, 2]
    for i, k in enumerate(ks):
        bf.insert(f"x{i}", k)
    assert bf.total_insertions == sum(ks)


def test_query_rejects_invalid_k():
    bf = BloomFilter(num_bits=100, seed=0)
    with pytest.raises(ValueError):
        bf.insert("x", k=0)


def test_empirical_fpr_matches_theory_within_tolerance():
    """Insert n items with a fixed k into a filter of size m, then measure
    the false positive rate on a large sample of items NOT inserted. This
    should match the closed-form independence approximation reasonably
    closely (it is an approximation, so we allow generous relative
    tolerance rather than exact equality)."""
    rng = np.random.default_rng(7)
    m = 20_000
    n = 2_000
    k = 5

    bf = BloomFilter(num_bits=m, seed=3)
    members = [f"member-{i}" for i in range(n)]
    for item in members:
        bf.insert(item, k)

    predicted_load = load_factor(m, bf.total_insertions)
    predicted_fpr = item_fpr(predicted_load, k)

    trials = 20_000
    negatives = [f"absent-{i}" for i in range(trials)]
    false_positives = sum(bf.query(item, k) for item in negatives)
    empirical_fpr = false_positives / trials

    # Loose tolerance: independence approximation + finite-sample noise.
    assert empirical_fpr == pytest.approx(predicted_fpr, rel=0.35, abs=0.01)
    # Sanity: also matches the measured bit load factor directly.
    assert bf.load_factor() == pytest.approx(predicted_load, rel=0.15)
