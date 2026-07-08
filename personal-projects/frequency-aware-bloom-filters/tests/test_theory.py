import pytest

from fab.theory import expected_weighted_fpr, item_fpr, load_factor


def test_load_factor_bounds():
    assert load_factor(num_bits=1000, total_insertions=0) == pytest.approx(0.0)
    # As insertions -> infinity, load factor -> 1.
    assert load_factor(num_bits=1000, total_insertions=1_000_000) > 0.999


def test_item_fpr_decreases_with_k():
    load = 0.5
    fprs = [item_fpr(load, k) for k in range(1, 10)]
    assert all(earlier > later for earlier, later in zip(fprs, fprs[1:]))


def test_item_fpr_increases_with_load():
    k = 5
    loads = [0.1, 0.3, 0.5, 0.7, 0.9]
    fprs = [item_fpr(load, k) for load in loads]
    assert all(earlier < later for earlier, later in zip(fprs, fprs[1:]))


def test_expected_weighted_fpr_matches_uniform_case():
    """When every key has k=k_base, the weighted average must equal the
    plain per-item FPR regardless of the weight distribution."""
    load = 0.6
    k_base = 5
    weights = {f"k{i}": float(i + 1) for i in range(10)}
    ks = {key: k_base for key in weights}
    result = expected_weighted_fpr(weights, ks, load)
    assert result == pytest.approx(item_fpr(load, k_base))


def test_weighted_allocation_beats_uniform_under_skew():
    """Core theoretical claim: shifting hash-function budget toward
    high-weight keys strictly lowers expected FPR versus a uniform
    allocation with the same average k, when weights are skewed."""
    load = 1 - pow(2.718281828, -0.7)  # matches T/m = 0.7 in the experiments
    k_base = 7.0

    weights = {f"k{i}": 2.0 ** (-i) for i in range(20)}  # geometric skew

    uniform_ks = {key: 7 for key in weights}
    uniform_fpr = expected_weighted_fpr(weights, uniform_ks, load)

    # Two-tier: top 10% get k=10, rest get the budget-matched k_cold.
    hot = set(sorted(weights, key=weights.get, reverse=True)[:2])  # top 10% of 20
    k_hot, hot_fraction = 10, 0.1
    k_cold = (k_base - hot_fraction * k_hot) / (1 - hot_fraction)
    weighted_ks = {key: (k_hot if key in hot else k_cold) for key in weights}

    def fpr_with_float_k(weights, ks, load):
        total_w = sum(weights.values())
        return sum(weights[k] * (load ** ks[k]) for k in ks) / total_w

    weighted_fpr = fpr_with_float_k(weights, weighted_ks, load)

    assert weighted_fpr < uniform_fpr
