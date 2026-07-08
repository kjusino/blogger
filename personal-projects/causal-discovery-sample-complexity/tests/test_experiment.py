import numpy as np

from src.experiment import (
    bonferroni_alpha,
    max_cond_set_cap,
    min_n_for_power,
    run_trial,
    recovery_probability,
    run_config,
    locate_n50,
    fine_grid_around,
)


def test_bonferroni_alpha_shrinks_with_p():
    a10 = bonferroni_alpha(10)
    a100 = bonferroni_alpha(100)
    assert a100 < a10
    assert np.isclose(a10, 0.05 / 45)


def test_max_cond_set_cap_bounded_and_scales_with_degree():
    assert max_cond_set_cap(p=50, d=2) == 5
    assert max_cond_set_cap(p=6, d=4) == 4  # capped at p - 2


def test_min_n_for_power_increases_as_alpha_shrinks():
    n_loose = min_n_for_power(p=10, alpha=0.01, min_margin=0.1)
    n_strict = min_n_for_power(p=10, alpha=1e-6, min_margin=0.1)
    assert n_strict > n_loose


def test_run_trial_returns_consistent_result():
    rng = np.random.default_rng(0)
    result = run_trial(p=8, d=2, n=5000, alpha=0.01, min_margin=0.1, rng=rng)
    assert result.p == 8 and result.d == 2 and result.n == 5000
    assert result.shd >= 0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert result.realized_degree <= 2


def test_recovery_probability_increases_with_sample_size():
    rng = np.random.default_rng(1)
    alpha = bonferroni_alpha(8)
    low = recovery_probability(p=8, d=2, n=30, trials=12, alpha=alpha, min_margin=0.1, rng=rng)
    high = recovery_probability(p=8, d=2, n=8000, trials=12, alpha=alpha, min_margin=0.1, rng=rng)
    assert high["recovery_prob"] >= low["recovery_prob"]
    assert high["mean_shd"] <= low["mean_shd"]


def test_locate_n50_returns_value_within_grid_bounds():
    rng = np.random.default_rng(2)
    alpha = bonferroni_alpha(8)
    n50, grid, probs = locate_n50(p=8, d=1, alpha=alpha, min_margin=0.1, rng=rng, coarse_trials=6, coarse_points=8, n_hi=20000)
    assert grid[0] <= n50 <= grid[-1]
    assert len(probs) == len(grid)


def test_fine_grid_around_respects_floor_and_spans_center():
    grid = fine_grid_around(center=1000, n_lo_floor=50, points=5, span=4.0)
    assert grid.min() >= 50
    assert grid.max() <= 4000
    assert len(grid) <= 5


def test_run_config_end_to_end_small():
    rng = np.random.default_rng(3)
    result = run_config(p=8, d=1, rng=rng, coarse_trials=6, fine_trials=10, fine_points=5, min_margin=0.1)
    assert result["p"] == 8 and result["d"] == 1
    assert result["n50_interp"] > 0
    assert len(result["rows"]) >= 1
    for row in result["rows"]:
        assert 0.0 <= row["recovery_prob"] <= 1.0
        assert row["trials"] == 10
