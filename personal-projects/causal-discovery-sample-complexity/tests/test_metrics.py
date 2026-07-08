import numpy as np

from src.metrics import (
    skeleton_shd,
    skeleton_precision_recall,
    exact_recovery,
    fit_logistic_threshold,
    n50_via_interpolation,
    loglog_slope,
)


def test_skeleton_shd_zero_for_identical_graphs():
    adj = np.array([[False, True], [True, False]])
    assert skeleton_shd(adj, adj) == 0


def test_skeleton_shd_counts_one_per_differing_pair():
    a = np.array([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    b = np.array([
        [False, True, True],
        [True, False, False],
        [True, False, False],
    ])
    # differ only on pair (0, 2)
    assert skeleton_shd(a, b) == 1


def test_exact_recovery_true_iff_shd_zero():
    a = np.array([[False, True], [True, False]])
    b = np.array([[False, False], [False, False]])
    assert exact_recovery(a, a) is True
    assert exact_recovery(a, b) is False


def test_precision_recall_perfect_match():
    adj = np.array([
        [False, True, False],
        [True, False, True],
        [False, True, False],
    ])
    p, r = skeleton_precision_recall(adj, adj)
    assert p == 1.0 and r == 1.0


def test_precision_recall_extra_edge_lowers_precision_only():
    true = np.array([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    est = np.array([
        [False, True, True],
        [True, False, False],
        [True, False, False],
    ])
    p, r = skeleton_precision_recall(est, true)
    assert p == 0.5
    assert r == 1.0


def test_precision_recall_missing_edge_lowers_recall_only():
    true = np.array([
        [False, True, False],
        [True, False, False],
        [False, False, False],
    ])
    est = np.zeros((3, 3), dtype=bool)
    p, r = skeleton_precision_recall(est, true)
    assert p == 1.0  # vacuously, no positives predicted
    assert r == 0.0


def test_n50_via_interpolation_exact_crossing():
    xs = np.array([10, 20, 30, 40])
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    n50 = n50_via_interpolation(xs, probs)
    assert 20 <= n50 <= 30


def test_n50_via_interpolation_linear_interpolation_value():
    xs = np.array([10, 20])
    probs = np.array([0.4, 0.6])
    n50 = n50_via_interpolation(xs, probs)
    assert np.isclose(n50, 15.0)


def test_n50_via_interpolation_all_above_half():
    xs = np.array([10, 20, 30])
    probs = np.array([0.6, 0.7, 0.9])
    assert n50_via_interpolation(xs, probs) == 10


def test_n50_via_interpolation_all_below_half():
    xs = np.array([10, 20, 30])
    probs = np.array([0.1, 0.2, 0.3])
    assert n50_via_interpolation(xs, probs) == 30


def test_fit_logistic_threshold_recovers_known_crossing():
    x0_true, k_true = 100.0, 0.05
    xs = np.linspace(10, 200, 20)
    probs = 1.0 / (1.0 + np.exp(-k_true * (xs - x0_true)))
    x0_fit, k_fit = fit_logistic_threshold(xs, probs)
    assert abs(x0_fit - x0_true) < 5.0
    assert k_fit > 0


def test_loglog_slope_recovers_known_power_law():
    x = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    y = 3.0 * x ** 2.0
    result = loglog_slope(x, y)
    assert np.isclose(result["slope"], 2.0, atol=1e-6)
    assert result["r_squared"] > 0.999
