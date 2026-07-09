import numpy as np

from src.experiment import algorithm_comparison, exp3_scaling_over_K, exp3_scaling_over_T
from src.regret import exp3_bound, fit_power_law


def test_exp3_scaling_over_T_end_to_end_shape():
    records = exp3_scaling_over_T(n_arms=6, t_values=[500, 1000, 2000], delta=0.3, n_seeds=4, base_seed=11)
    assert len(records) == 3 * 4
    for r in records:
        assert r["regret"] >= 0.0
        assert not np.isnan(r["regret"])


def test_exp3_scaling_over_T_fitted_exponent_is_near_theoretical_half():
    # Small but real grid: sanity-checks the whole pipeline (env -> algo ->
    # regret -> fit) end to end, not just that it runs without crashing.
    records = exp3_scaling_over_T(
        n_arms=6, t_values=[500, 1000, 2000, 4000, 8000], delta=0.3, n_seeds=6, base_seed=12
    )
    by_t = {}
    for r in records:
        by_t.setdefault(r["T"], []).append(r["regret"])
    t_values = sorted(by_t)
    mean_regret = [np.mean(by_t[t]) for t in t_values]
    exponent, r_squared = fit_power_law(t_values, mean_regret)
    assert 0.3 < exponent < 0.7
    assert r_squared > 0.8


def test_exp3_scaling_over_K_end_to_end_shape():
    records = exp3_scaling_over_K(k_values=[2, 4, 8], horizon=2000, delta=0.3, n_seeds=4, base_seed=13)
    assert len(records) == 3 * 4
    for r in records:
        assert r["regret"] >= 0.0


def test_exp3_regret_usually_within_a_small_multiple_of_its_bound():
    # The bound is an *expectation* bound with a generous constant, so a
    # single seed can exceed it; the aggregate (median across seeds,
    # T) should not exceed a very generous slack multiple.
    records = exp3_scaling_over_T(n_arms=6, t_values=[1000, 4000], delta=0.3, n_seeds=8, base_seed=14)
    ratios = []
    for r in records:
        bound = exp3_bound(r["K"], r["T"])
        ratios.append(r["regret"] / bound)
    assert np.median(ratios) < 1.0  # comfortably inside the bound on average
    assert max(ratios) < 3.0  # generous slack for a worst single seed


def test_algorithm_comparison_stochastic_end_to_end():
    records = algorithm_comparison(
        "stochastic", n_arms=5, t_values=[500, 2000], delta=0.3, n_seeds=4, base_seed=15
    )
    assert len(records) == 2 * 2 * 4
    algos = {r["algo"] for r in records}
    assert algos == {"EXP3", "UCB1"}
    for r in records:
        assert r["regret"] >= 0.0


def test_algorithm_comparison_switching_end_to_end():
    records = algorithm_comparison(
        "switching", n_arms=5, t_values=[500, 2000], delta=0.3, n_seeds=4, base_seed=16, num_segments=4
    )
    assert len(records) == 2 * 2 * 4
    for r in records:
        # Dynamic (per-round-best-arm) pseudo-regret is a sum of
        # nonnegative per-round gaps, so it can never go negative.
        assert r["regret"] >= 0.0


def test_ucb1_beats_exp3_in_stationary_stochastic_regime():
    # The whole point of the comparison: UCB1 should have lower regret
    # than EXP3 once T is large enough for the log(T) vs sqrt(T) gap to
    # dominate the constants.
    records = algorithm_comparison(
        "stochastic", n_arms=5, t_values=[8000], delta=0.3, n_seeds=10, base_seed=17
    )
    ucb1_mean = np.mean([r["regret"] for r in records if r["algo"] == "UCB1"])
    exp3_mean = np.mean([r["regret"] for r in records if r["algo"] == "EXP3"])
    assert ucb1_mean < exp3_mean


def test_ucb1_dynamic_regret_does_not_exceed_exp3_even_under_switching():
    # Counterintuitive empirical finding this project is built around: a
    # UCB1 that has no non-stationarity guarantee at all still matches or
    # beats EXP3's *guaranteed-robust* dynamic regret across the whole
    # tested range of switching frequencies, converging to (not past)
    # EXP3's level only once segments shrink to ~K (see README).
    for num_segments in (4, 32, 8000 // 5):
        records = algorithm_comparison(
            "switching", n_arms=5, t_values=[8000], delta=0.3, n_seeds=8, base_seed=19 + num_segments,
            num_segments=num_segments,
        )
        ucb1_mean = np.mean([r["regret"] for r in records if r["algo"] == "UCB1"])
        exp3_mean = np.mean([r["regret"] for r in records if r["algo"] == "EXP3"])
        assert ucb1_mean <= exp3_mean * 1.05  # small slack for seed noise


def test_ucb1_switching_regret_grows_toward_exp3_level_as_segments_shorten():
    long_segments = algorithm_comparison(
        "switching", n_arms=5, t_values=[8000], delta=0.3, n_seeds=8, base_seed=41, num_segments=4
    )
    short_segments = algorithm_comparison(
        "switching", n_arms=5, t_values=[8000], delta=0.3, n_seeds=8, base_seed=42, num_segments=8000 // 5
    )
    ucb1_long = np.mean([r["regret"] for r in long_segments if r["algo"] == "UCB1"])
    ucb1_short = np.mean([r["regret"] for r in short_segments if r["algo"] == "UCB1"])
    assert ucb1_long < ucb1_short
