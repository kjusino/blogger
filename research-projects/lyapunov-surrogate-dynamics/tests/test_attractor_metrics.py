import numpy as np

from src.attractor_metrics import marginal_js_divergence, DEFAULT_RANGES


def test_js_divergence_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    traj = rng.uniform(low=DEFAULT_RANGES[:, 0], high=DEFAULT_RANGES[:, 1],
                        size=(5000, 3))
    result = marginal_js_divergence(traj, traj.copy())
    assert result["mean"] < 1e-6
    assert np.all(result["per_coord"] < 1e-6)


def test_js_divergence_positive_for_visibly_different_distributions():
    rng = np.random.default_rng(1)
    # Two clearly different distributions: one concentrated near the low
    # end of each coordinate's range, one near the high end.
    lo = DEFAULT_RANGES[:, 0] + 0.05 * (DEFAULT_RANGES[:, 1] - DEFAULT_RANGES[:, 0])
    hi = DEFAULT_RANGES[:, 1] - 0.05 * (DEFAULT_RANGES[:, 1] - DEFAULT_RANGES[:, 0])
    traj_a = lo + rng.normal(0, 0.5, size=(3000, 3))
    traj_b = hi + rng.normal(0, 0.5, size=(3000, 3))

    result = marginal_js_divergence(traj_a, traj_b)
    # JS divergence (base-e) is bounded above by ln(2) ~= 0.693; two
    # non-overlapping distributions should sit close to that bound.
    assert result["mean"] > 0.5
    assert np.all(result["per_coord"] > 0.4)


def test_js_divergence_between_zero_and_identical_case_is_ordered():
    rng = np.random.default_rng(2)
    traj_true = rng.normal(loc=0.0, scale=5.0, size=(4000, 3))
    traj_same = traj_true + rng.normal(0, 0.01, size=(4000, 3))  # near-identical
    traj_diff = rng.normal(loc=15.0, scale=5.0, size=(4000, 3))  # shifted

    d_same = marginal_js_divergence(traj_true, traj_same)["mean"]
    d_diff = marginal_js_divergence(traj_true, traj_diff)["mean"]
    assert d_same < d_diff
