import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.experiment import (
    run_single,
    run_sweep_B,
    run_sweep_M,
    run_sweep_n,
    run_sweep_naive_capacity_cliff,
    run_sweep_tall_cache_boundary,
)


def test_run_single_is_deterministic_given_seed():
    r1 = run_single(16, 4, 256, "oblivious", seed=42)
    r2 = run_single(16, 4, 256, "oblivious", seed=42)
    assert r1["misses"] == r2["misses"]
    assert r1["hits"] == r2["hits"]


def test_run_single_different_seeds_still_valid_result_shape():
    r = run_single(16, 4, 256, "naive", seed=1)
    assert r["misses"] > 0
    assert r["hits"] > 0
    assert r["total_accesses"] == r["hits"] + r["misses"]
    assert r["n"] == 16 and r["B"] == 4 and r["M"] == 256
    assert r["algorithm"] == "naive"


def test_run_single_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        run_single(8, 4, 64, "not-an-algorithm")


def test_run_sweep_n_produces_one_record_per_n_per_algorithm():
    n_values = [8, 16, 24]
    algos = ["naive", "oblivious"]
    records = run_sweep_n(n_values, B=4, M=256, algorithms=algos)
    assert len(records) == len(n_values) * len(algos)
    seen = {(r["algorithm"], r["n"]) for r in records}
    assert seen == {(a, n) for a in algos for n in n_values}
    assert all(r["experiment"] == "scaling_n" for r in records)


def test_run_sweep_B_and_M_shapes():
    b_records = run_sweep_B(n=16, B_values=[2, 4, 8], M=256, algorithms=["blocked"])
    assert len(b_records) == 3
    assert all(r["experiment"] == "scaling_B" for r in b_records)

    m_records = run_sweep_M(n=16, B=4, M_values=[64, 128], algorithms=["blocked", "oblivious"])
    assert len(m_records) == 4
    assert all(r["experiment"] == "scaling_M" for r in m_records)


def test_tall_cache_sweep_computes_ratio():
    records = run_sweep_tall_cache_boundary(n=16, B=4, M_values=[16, 32, 64], algorithms=["oblivious"])
    for r in records:
        assert r["tall_cache_ratio"] == pytest.approx(r["M"] / 16)  # B^2 = 16


def test_naive_capacity_cliff_sweep_only_runs_naive_and_tags_ratio():
    threshold = 16 * 4  # n * B
    records = run_sweep_naive_capacity_cliff(n=16, B=4, M_values=[32, threshold, 128])
    assert all(r["algorithm"] == "naive" for r in records)
    assert all(r["experiment"] == "naive_capacity_cliff" for r in records)
    below = next(r for r in records if r["M"] < threshold)
    above = next(r for r in records if r["M"] >= threshold)
    assert below["capacity_ratio"] < 1.0
    assert above["capacity_ratio"] >= 1.0


def test_naive_capacity_cliff_shows_a_real_drop():
    # weak, robust regression test: crossing the M = n*B threshold should
    # substantially reduce naive's miss count, regardless of exact constants.
    n, B = 32, 4
    threshold = n * B
    records = run_sweep_naive_capacity_cliff(n, B, M_values=[threshold // 4, threshold * 4])
    below = next(r for r in records if r["M"] == threshold // 4)
    above = next(r for r in records if r["M"] == threshold * 4)
    assert above["misses"] < below["misses"] / 2
