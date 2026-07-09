import numpy as np

from src.stats_utils import linear_fit, mean_ci, two_sample_diff_ci


def test_mean_ci_on_constant_samples_is_a_point():
    mean, lo, hi = mean_ci([0.7] * 50)
    assert abs(mean - 0.7) < 1e-9
    assert abs(lo - 0.7) < 1e-9
    assert abs(hi - 0.7) < 1e-9


def test_mean_ci_bracket_contains_true_mean_with_high_frequency():
    rng = np.random.default_rng(0)
    hits = 0
    trials = 200
    for _ in range(trials):
        samples = rng.uniform(0.4, 0.9, size=60)
        mean, lo, hi = mean_ci(samples, rng=rng, n_bootstrap=500)
        if lo <= mean <= hi:
            hits += 1
    assert hits == trials  # the CI must always contain its own point estimate


def test_mean_ci_empty_returns_nan():
    mean, lo, hi = mean_ci([])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)


def test_two_sample_diff_ci_detects_a_clear_separation():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.8, 0.95, size=100)  # clearly higher
    b = rng.uniform(0.5, 0.65, size=100)  # clearly lower
    diff, lo, hi = two_sample_diff_ci(a, b, rng=rng)
    assert diff > 0
    assert lo > 0  # CI excludes zero -> significant


def test_two_sample_diff_ci_is_inconclusive_for_identical_distributions():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.5, 0.9, size=200)
    b = rng.uniform(0.5, 0.9, size=200)
    diff, lo, hi = two_sample_diff_ci(a, b, rng=rng)
    assert lo < 0 < hi  # CI should straddle zero for draws from the same distribution


def test_linear_fit_recovers_exact_line():
    x = [0, 1, 2, 3, 4]
    y = [2, 4, 6, 8, 10]  # y = 2x + 2
    slope, intercept, r2 = linear_fit(x, y)
    assert abs(slope - 2.0) < 1e-9
    assert abs(intercept - 2.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9
