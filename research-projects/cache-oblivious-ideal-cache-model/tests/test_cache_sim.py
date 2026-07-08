import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.cache_sim import Array1D, IdealCache, Matrix


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        IdealCache(M=0, B=4)
    with pytest.raises(ValueError):
        IdealCache(M=4, B=0)
    with pytest.raises(ValueError):
        IdealCache(M=2, B=4)  # M < B: can't even hold one block


def test_capacity_in_lines():
    cache = IdealCache(M=64, B=8)
    assert cache.L == 8


def test_compulsory_miss_then_hit_within_block():
    cache = IdealCache(M=1024, B=8)
    assert cache.access(0) is False  # cold miss
    for addr in range(1, 8):
        assert cache.access(addr) is True  # same block, all hits
    assert cache.misses == 1
    assert cache.hits == 7


def test_sequential_scan_misses_equal_ceil_n_over_b():
    B = 8
    n = 100
    cache = IdealCache(M=4096, B=B)
    for addr in range(n):
        cache.access(addr)
    expected_misses = -(-n // B)  # ceil division
    assert cache.misses == expected_misses
    assert cache.hits == n - expected_misses


def test_lru_eviction_order():
    # capacity 2 lines, block size 1 word -> capacity 2 words
    cache = IdealCache(M=2, B=1)
    cache.access(0)  # miss, resident: [0]
    cache.access(1)  # miss, resident: [0,1]
    cache.access(0)  # hit, resident: [1,0] (0 now most recent)
    cache.access(2)  # miss, evicts LRU = 1; resident: [0,2]
    assert cache.access(1) is False  # 1 was evicted -> miss
    assert cache.access(2) is True  # 2 still resident -> hit


def test_fully_resident_dataset_only_compulsory_misses():
    B = 4
    n_words = 64
    cache = IdealCache(M=n_words, B=B)  # exactly fits
    for _ in range(5):
        for addr in range(n_words):
            cache.access(addr)
    expected_compulsory = n_words // B
    assert cache.misses == expected_compulsory
    assert cache.hits == 5 * n_words - expected_compulsory


def test_array1d_shares_cache_state_with_offset():
    cache = IdealCache(M=1024, B=8)
    a = Array1D(cache, base_address=0, size=16)
    b = Array1D(cache, base_address=16, size=16)
    a.get(0)  # touches block 0
    assert cache.misses == 1
    b.get(0)  # touches block for address 16, a distinct block
    assert cache.misses == 2
    a.get(1)  # same block as a.get(0) -> hit
    assert cache.hits == 1


def test_matrix_row_major_indexing_matches_manual_offset():
    cache = IdealCache(M=1024, B=8)
    m = Matrix(cache, base_address=0, n=4)
    m.set(2, 3, 42.0)
    assert m.arr.data[2 * 4 + 3] == 42.0
    assert m.get(2, 3) == 42.0


def test_matrix_from_list_and_to_list_roundtrip():
    cache = IdealCache(M=1024, B=8)
    rows = [[1.0, 2.0], [3.0, 4.0]]
    m = Matrix.from_list(cache, 0, rows)
    assert m.to_list() == rows


def test_reset_counters():
    cache = IdealCache(M=64, B=8)
    cache.access(0)
    cache.access(1000)
    assert cache.hits + cache.misses > 0
    cache.reset_counters()
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.total_accesses == 0
