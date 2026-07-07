import math

import numpy as np

from src.audit import (
    audit_roc,
    best_audit_epsilon,
    clopper_pearson_fpr_upper,
    clopper_pearson_tpr_lower,
    eps_lower_from_counts,
    sweep_thresholds,
)


def test_perfect_separation_gives_large_finite_eps_lower():
    n_in, n_out = 200, 200
    eps = eps_lower_from_counts(tp=n_in, n_in=n_in, fp=0, n_out=n_out, delta=1e-5)
    assert math.isfinite(eps)
    assert eps > 3.0  # perfect separation at N=200 should give a solidly large bound


def test_no_separation_gives_eps_close_to_zero():
    n_in, n_out = 200, 200
    tp = n_in // 2
    fp = n_out // 2
    eps = eps_lower_from_counts(tp=tp, n_in=n_in, fp=fp, n_out=n_out, delta=1e-5)
    assert eps < 0.5


def test_eps_lower_monotonic_nondecreasing_in_tp():
    n_in, n_out = 100, 100
    fp = 20
    prev = -math.inf
    for tp in range(0, n_in + 1, 5):
        eps = eps_lower_from_counts(tp=tp, n_in=n_in, fp=fp, n_out=n_out, delta=1e-5)
        assert eps >= prev - 1e-9  # non-decreasing (allow tiny float slack)
        prev = eps


def test_clopper_pearson_bounds_basic_properties():
    assert clopper_pearson_tpr_lower(0, 100) == 0.0
    assert clopper_pearson_fpr_upper(100, 100) == 1.0
    # More true positives -> higher (or equal) TPR lower bound.
    assert clopper_pearson_tpr_lower(90, 100) > clopper_pearson_tpr_lower(50, 100)
    # More false positives -> higher (or equal) FPR upper bound.
    assert clopper_pearson_fpr_upper(50, 100) > clopper_pearson_fpr_upper(5, 100)


def test_eps_lower_never_negative():
    for tp in [0, 10, 50, 100]:
        for fp in [0, 10, 50, 100]:
            eps = eps_lower_from_counts(tp=tp, n_in=100, fp=fp, n_out=100, delta=1e-5)
            assert eps >= 0.0


def test_sweep_thresholds_are_midpoints_of_unique_values():
    stats_in = np.array([1.0, 2.0, 3.0])
    stats_out = np.array([2.0, 4.0])
    thresholds = sweep_thresholds(stats_in, stats_out)
    uniq = np.array([1.0, 2.0, 3.0, 4.0])
    expected = (uniq[:-1] + uniq[1:]) / 2.0
    np.testing.assert_allclose(thresholds, expected)


def test_audit_roc_and_best_epsilon_on_clearly_separated_synthetic_stats():
    rng = np.random.default_rng(0)
    stats_in = rng.normal(loc=-5.0, scale=0.5, size=100)
    stats_out = rng.normal(loc=5.0, scale=0.5, size=100)

    roc = audit_roc(stats_in, stats_out)
    assert len(roc) > 0
    for point in roc:
        assert 0.0 <= point["tpr"] <= 1.0
        assert 0.0 <= point["fpr"] <= 1.0

    best = best_audit_epsilon(stats_in, stats_out, delta=1e-5)
    assert best["eps_lower"] > 3.0
