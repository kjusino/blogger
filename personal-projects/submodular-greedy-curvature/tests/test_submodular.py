import itertools

import numpy as np
import pytest

from src.submodular import WeightedCoverageFunction, make_grouped_redundancy_instance


def make_instance(n=10, m=25, p=0.25, seed=0):
    rng = np.random.default_rng(seed)
    return WeightedCoverageFunction(n, m, p, weight_low=1.0, weight_high=10.0, rng=rng)


@pytest.mark.parametrize("seed", range(5))
def test_monotonicity(seed):
    """f(S) <= f(T) whenever S subseteq T."""
    f = make_instance(seed=seed)
    rng = np.random.default_rng(seed + 1000)
    elements = list(range(f.n))
    rng.shuffle(elements)
    running = set()
    prev_val = 0.0
    for e in elements:
        running.add(e)
        val = f.value(running)
        assert val >= prev_val - 1e-9
        prev_val = val


@pytest.mark.parametrize("seed", range(5))
def test_diminishing_returns(seed):
    """Submodularity: for S subseteq T and e not in T, f(e|S) >= f(e|T)."""
    f = make_instance(seed=seed)
    rng = np.random.default_rng(seed + 2000)
    all_elems = list(range(f.n))
    rng.shuffle(all_elems)
    split = f.n // 2
    S = set(all_elems[:split])
    T = set(all_elems[: split + 2]) if split + 2 <= f.n else set(all_elems)
    assert S.issubset(T)
    for e in range(f.n):
        if e in T:
            continue
        gain_S = f.marginal_gain(e, S)
        gain_T = f.marginal_gain(e, T)
        assert gain_S >= gain_T - 1e-9, (
            f"submodularity violated: f({e}|S)={gain_S} < f({e}|T)={gain_T}"
        )


def test_value_of_empty_set_is_zero():
    f = make_instance()
    assert f.value(set()) == 0.0
    assert f.value(frozenset()) == 0.0


def test_value_of_full_set_matches_union_of_all_features():
    f = make_instance(n=6, m=15, p=0.3, seed=42)
    covered_mask = np.zeros(f.m, dtype=bool)
    for i in range(f.n):
        covered_mask |= f.covers[i]
    expected = float(f.feature_weights[covered_mask].sum())
    assert f.value(set(range(f.n))) == pytest.approx(expected)


def test_marginal_gain_matches_value_difference():
    f = make_instance(seed=7)
    S = {0, 2, 4}
    for j in range(f.n):
        expected = f.value(S | {j}) - f.value(S)
        assert f.marginal_gain(j, S) == pytest.approx(expected, abs=1e-9)


def test_marginal_gain_of_element_already_in_set_is_zero():
    f = make_instance(seed=3)
    S = {1, 2, 3}
    assert f.marginal_gain(2, S) == 0.0


def test_curvature_is_in_unit_interval():
    for seed in range(10):
        f = make_instance(seed=seed, p=0.3)
        c = f.curvature()
        assert -1e-9 <= c <= 1.0 + 1e-9


def test_curvature_near_zero_for_disjoint_coverage():
    """When elements cover disjoint feature blocks (no overlap possible),
    the function is exactly modular/additive, so curvature must be exactly 0:
    every element's marginal gain is the same regardless of context."""
    n, block = 5, 4
    m = n * block
    rng = np.random.default_rng(11)
    f = WeightedCoverageFunction(n, m, p=1e-9, weight_low=1.0, weight_high=1.0, rng=rng)
    # Force a clean disjoint-block structure by hand (bypass random overlap).
    f.covers[:, :] = False
    for i in range(n):
        f.covers[i, i * block:(i + 1) * block] = True
    f.feature_weights = np.ones(m)

    c = f.curvature()
    assert c == pytest.approx(0.0, abs=1e-9)


def test_curvature_near_one_for_fully_redundant_coverage():
    """If every element covers the exact same single feature, adding any
    element after another gives zero marginal value: curvature -> 1."""
    n, m = 6, 3
    rng = np.random.default_rng(5)
    f = WeightedCoverageFunction(n, m, p=1e-9, weight_low=1.0, weight_high=1.0, rng=rng)
    f.covers[:, :] = False
    f.covers[:, 0] = True  # every element covers only feature 0
    f.feature_weights = np.ones(m)

    c = f.curvature()
    assert c == pytest.approx(1.0, abs=1e-9)


def test_rejects_invalid_p():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        WeightedCoverageFunction(5, 5, p=0.0, weight_low=1.0, weight_high=2.0, rng=rng)
    with pytest.raises(ValueError):
        WeightedCoverageFunction(5, 5, p=1.5, weight_low=1.0, weight_high=2.0, rng=rng)


def make_grouped(n=14, n_groups=10, group_prob=0.35, mult=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return make_grouped_redundancy_instance(n, n_groups, group_prob, mult, 1.0, 10.0, rng)


def test_grouped_redundancy_zero_mult_is_exactly_modular_curvature_zero():
    """With redundancy_mult=0, only private (exclusive) features exist, so
    the function is exactly modular: curvature must be exactly 0."""
    for seed in range(10):
        f = make_grouped(mult=0.0, seed=seed)
        assert f.curvature() == pytest.approx(0.0, abs=1e-9)
        for j in range(f.n):
            alone = f.marginal_gain(j, set())
            in_context = f.marginal_gain(j, set(range(f.n)) - {j})
            assert alone == pytest.approx(in_context, abs=1e-9)


def test_grouped_redundancy_zero_mult_greedy_is_exactly_optimal():
    """A modular function's cardinality-constrained optimum is trivially
    the k highest-weight elements, which greedy always finds exactly."""
    from src.greedy import greedy
    from src.optimal import brute_force_opt

    f = make_grouped(n=9, mult=0.0, seed=3)
    for k in range(1, f.n + 1):
        opt_val, _ = brute_force_opt(f, k)
        _, trace = greedy(f, k)
        assert trace[-1] == pytest.approx(opt_val, rel=1e-6)


def test_grouped_redundancy_curvature_increases_with_mult_on_average():
    """Higher redundancy_mult should push curvature up (on average across
    seeds), spanning a real range rather than saturating immediately."""
    mults = [0.0, 0.1, 1.0, 10.0]
    mean_curvatures = []
    for mult in mults:
        cs = [make_grouped(mult=mult, seed=s).curvature() for s in range(15)]
        mean_curvatures.append(np.mean(cs))
    assert mean_curvatures[0] == pytest.approx(0.0, abs=1e-9)
    assert all(a < b for a, b in zip(mean_curvatures, mean_curvatures[1:]))
    # a genuine spread, not everything crammed against 1
    assert mean_curvatures[1] < 0.9


def test_grouped_redundancy_is_monotone_and_submodular():
    for seed in range(5):
        f = make_grouped(mult=1.5, seed=seed)
        running, prev = set(), 0.0
        order = list(range(f.n))
        for e in order:
            running.add(e)
            v = f.value(running)
            assert v >= prev - 1e-9
            prev = v
        S, T = {0, 1, 2}, {0, 1, 2, 3, 4}
        for e in range(f.n):
            if e in T:
                continue
            assert f.marginal_gain(e, S) >= f.marginal_gain(e, T) - 1e-9


def test_grouped_redundancy_rejects_invalid_params():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        make_grouped_redundancy_instance(1, 5, 0.3, 1.0, 1.0, 10.0, rng)
    with pytest.raises(ValueError):
        make_grouped_redundancy_instance(5, 5, 0.3, -1.0, 1.0, 10.0, rng)


def test_grouped_redundancy_every_group_has_at_least_two_members():
    """Small n and low group_membership_prob would otherwise often produce
    singleton (non-redundant) groups; construction must force >=2 members."""
    rng = np.random.default_rng(1)
    f = make_grouped_redundancy_instance(5, 20, 0.05, 2.0, 1.0, 10.0, rng)
    group_cols = f.covers[:, 5:]
    assert (group_cols.sum(axis=0) >= 2).all()


def test_every_element_has_positive_standalone_value():
    """Construction must guarantee f(j|{}) > 0 for all j (curvature's
    denominator would otherwise be undefined)."""
    for seed in range(20):
        f = make_instance(n=8, m=6, p=0.05, seed=seed)  # low p stresses the guard
        for j in range(f.n):
            assert f.marginal_gain(j, set()) > 0.0
