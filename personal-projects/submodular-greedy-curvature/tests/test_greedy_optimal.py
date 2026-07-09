import numpy as np
import pytest

from src.greedy import greedy
from src.optimal import brute_force_opt
from src.submodular import WeightedCoverageFunction
from src.theory import WORST_CASE_BOUND, curvature_bound


def make_instance(n=8, m=20, p=0.25, seed=0):
    rng = np.random.default_rng(seed)
    return WeightedCoverageFunction(n, m, p, weight_low=1.0, weight_high=10.0, rng=rng)


def test_greedy_selects_k_distinct_elements():
    f = make_instance()
    selected, trace = greedy(f, k=4)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert len(trace) == 4


def test_greedy_value_trace_is_nondecreasing():
    f = make_instance(seed=2)
    _, trace = greedy(f, k=6)
    assert all(b >= a - 1e-9 for a, b in zip(trace, trace[1:]))


def test_greedy_final_trace_value_matches_direct_evaluation():
    f = make_instance(seed=9)
    selected, trace = greedy(f, k=5)
    assert trace[-1] == pytest.approx(f.value(set(selected)))


def test_greedy_k_zero_selects_nothing():
    f = make_instance()
    selected, trace = greedy(f, k=0)
    assert selected == []
    assert trace == []


def test_greedy_k_exceeding_n_is_clamped():
    f = make_instance(n=5, m=10, seed=1)
    selected, _ = greedy(f, k=100)
    assert len(selected) == 5
    assert set(selected) == set(range(5))


def test_brute_force_matches_greedy_when_k_equals_n():
    """When k == n, both algorithms must pick everything, so they must agree
    exactly (the one case where greedy is trivially optimal)."""
    f = make_instance(n=6, m=15, seed=4)
    opt_val, _ = brute_force_opt(f, k=6)
    _, trace = greedy(f, k=6)
    assert trace[-1] == pytest.approx(opt_val)


def test_brute_force_never_beaten_by_greedy():
    """Sanity check on the brute-force oracle itself: no algorithm can beat
    the enumerated optimum."""
    for seed in range(8):
        f = make_instance(n=7, m=12, seed=seed)
        for k in (1, 3, 5):
            opt_val, _ = brute_force_opt(f, k)
            _, trace = greedy(f, k)
            assert trace[-1] <= opt_val + 1e-6


def test_greedy_beats_worst_case_bound_on_random_instances():
    """The foundational Nemhauser-Wolsey-Fisher guarantee: greedy achieves
    at least (1-1/e) * OPT. This must never be violated."""
    for seed in range(15):
        f = make_instance(n=8, m=16, p=0.3, seed=seed)
        for k in (2, 4, 6):
            opt_val, _ = brute_force_opt(f, k)
            _, trace = greedy(f, k)
            ratio = trace[-1] / opt_val
            assert ratio >= WORST_CASE_BOUND - 1e-6, (
                f"seed={seed} k={k}: ratio {ratio} below worst-case bound"
            )


def test_greedy_beats_curvature_bound_on_random_instances():
    """The tighter, curvature-refined guarantee (Conforti & Cornuejols 1984)
    must also never be violated: greedy >= (1-e^-c)/c * OPT."""
    for seed in range(15):
        f = make_instance(n=8, m=16, p=0.3, seed=seed)
        c = f.curvature()
        bound = curvature_bound(c)
        for k in (2, 4, 6):
            opt_val, _ = brute_force_opt(f, k)
            _, trace = greedy(f, k)
            ratio = trace[-1] / opt_val
            assert ratio >= bound - 1e-6, (
                f"seed={seed} k={k}: ratio {ratio} below curvature bound {bound} (c={c})"
            )


def test_greedy_is_exactly_optimal_on_disjoint_modular_instance():
    """Curvature 0 (purely additive/modular function) implies the curvature
    bound is exactly 1: greedy must be exactly optimal, not just close."""
    n, block = 5, 4
    m = n * block
    rng = np.random.default_rng(11)
    f = WeightedCoverageFunction(n, m, p=1e-9, weight_low=1.0, weight_high=1.0, rng=rng)
    f.covers[:, :] = False
    for i in range(n):
        f.covers[i, i * block:(i + 1) * block] = True
    f.feature_weights = rng.uniform(1.0, 10.0, size=m)

    for k in range(1, n + 1):
        opt_val, _ = brute_force_opt(f, k)
        _, trace = greedy(f, k)
        assert trace[-1] == pytest.approx(opt_val, rel=1e-6)
