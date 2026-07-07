import numpy as np
import pytest

from hypercube_cutoff import experiment as exp


def test_interp_crossing_basic():
    t = np.array([0, 10, 20, 30])
    tv = np.array([1.0, 0.8, 0.4, 0.1])
    crossing = exp.interp_crossing(t, tv, 0.6)
    # 0.6 is halfway between 0.8 (t=10) and 0.4 (t=20) -> t=15
    assert crossing == pytest.approx(15.0)


def test_interp_crossing_rejects_non_monotonic():
    t = np.array([0, 10, 20])
    tv = np.array([1.0, 1.2, 0.5])  # increases then decreases
    with pytest.raises(ValueError):
        exp.interp_crossing(t, tv, 0.6)


def test_interp_crossing_rejects_out_of_range_target():
    t = np.array([0, 10, 20])
    tv = np.array([0.9, 0.5, 0.2])
    with pytest.raises(ValueError):
        exp.interp_crossing(t, tv, 0.99)
    with pytest.raises(ValueError):
        exp.interp_crossing(t, tv, 0.01)


@pytest.mark.integration
def test_validate_lumping_agrees_across_methods():
    rows = exp.validate_lumping([8, 10], num_trials=5000, seed=0)
    assert len(rows) > 0
    max_diff = max(r["lumped_vs_bruteforce_abs_diff"] for r in rows)
    assert max_diff < 1e-8

    agreement_rate = sum(r["mc_within_ci"] for r in rows) / len(rows)
    assert agreement_rate > 0.7  # bootstrap CIs should usually contain the exact value


@pytest.mark.integration
def test_cutoff_scaling_sweep_relative_error_shrinks_with_n():
    n_values = [100, 400, 1600]
    c_values = np.linspace(-3, 5, 17)
    _, summary_rows = exp.cutoff_scaling_sweep(n_values, c_values)

    rel_errs = [r["rel_err_half_vs_cutoff_time"] for r in summary_rows]
    assert rel_errs[-1] < rel_errs[0]

    window_over_n = [r["window_over_n"] for r in summary_rows]
    assert max(window_over_n) - min(window_over_n) < 0.2  # roughly constant -> linear scaling


@pytest.mark.integration
def test_cutoff_scaling_sweep_self_collapse_shrinks_but_bound_gap_does_not():
    n_values = [100, 400, 1600]
    c_values = np.linspace(-3, 5, 17)
    _, summary_rows = exp.cutoff_scaling_sweep(n_values, c_values)

    self_errs = [r["self_collapse_error"] for r in summary_rows]
    assert self_errs[0] > self_errs[1] > 0  # shrinks toward the largest-n reference curve

    bound_gaps = [r["chi_square_bound_gap"] for r in summary_rows]
    # the classical chi-square bound is not asymptotically tight: the gap
    # stays roughly constant rather than shrinking like the self-collapse error
    assert abs(bound_gaps[0] - bound_gaps[-1]) < 0.05
    assert bound_gaps[-1] > 0.3


@pytest.mark.integration
def test_monte_carlo_validation_matches_exact_chain():
    n = 15
    c_values = np.linspace(-2, 4, 7)
    rows = exp.monte_carlo_validation(n, c_values, num_trials=15000, seed=1)
    within_ci = [r["bitvector_within_ci"] for r in rows]
    assert sum(within_ci) / len(within_ci) > 0.7
