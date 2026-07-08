import numpy as np
import pytest

from ramanujan_spectra.experiment import (
    CellSummary,
    TrialResult,
    cell_summary_to_row,
    compare_generators,
    fit_gap_power_law,
    make_rng,
    run_single_trial,
    run_sweep,
    summarize_cell,
    summarize_sweep,
    trial_result_to_row,
    trials_for_n,
)
from ramanujan_spectra.theory import alon_boppana_bound


def test_trials_for_n_schedule():
    assert trials_for_n(64) == 40
    assert trials_for_n(512) == 40
    assert trials_for_n(513) == 20
    assert trials_for_n(2048) == 20
    assert trials_for_n(2049) == 10
    assert trials_for_n(8192) == 10


def test_make_rng_is_deterministic():
    rng_a = make_rng(3, 100, 0, base_seed=42)
    rng_b = make_rng(3, 100, 0, base_seed=42)
    assert rng_a.integers(0, 10**9) == rng_b.integers(0, 10**9)


def test_make_rng_varies_with_trial():
    rng_a = make_rng(3, 100, 0, base_seed=42)
    rng_b = make_rng(3, 100, 1, base_seed=42)
    assert rng_a.integers(0, 10**9) != rng_b.integers(0, 10**9)


def test_run_single_trial_basic_sanity():
    r = run_single_trial(d=4, n=200, trial=0)
    assert r.d == 4
    assert r.n == 200
    assert r.lambda1 == pytest.approx(4.0, abs=1e-6)
    assert r.bound == pytest.approx(alon_boppana_bound(4))
    assert r.gap == pytest.approx(r.lambda2_abs - r.bound)
    # lambda(G) is, by construction, at most lambda1 = d
    assert r.lambda2_abs <= r.lambda1 + 1e-9


def test_run_sweep_produces_expected_count():
    results = run_sweep(degrees=(3, 4), n_grid=(50, 100), trials_fn=lambda n: 3)
    assert len(results) == 2 * 2 * 3
    assert {r.d for r in results} == {3, 4}
    assert {r.n for r in results} == {50, 100}


def _make_trial(d, n, trial, lambda2_abs):
    bound = alon_boppana_bound(d)
    return TrialResult(
        d=d,
        n=n,
        trial=trial,
        lambda1=float(d),
        lambda2=lambda2_abs,
        lambda_min=-lambda2_abs,
        lambda2_abs=lambda2_abs,
        bipartite_like=False,
        connected=True,
        bound=bound,
        gap=lambda2_abs - bound,
    )


def test_summarize_cell_matches_hand_computation():
    d, n = 4, 100
    bound = alon_boppana_bound(d)
    vals = [bound - 0.5, bound - 0.1, bound + 0.2]
    rows = [_make_trial(d, n, i, v) for i, v in enumerate(vals)]
    summary = summarize_cell(rows)

    assert summary.d == d
    assert summary.n == n
    assert summary.trials == 3
    assert summary.lambda2_abs_mean == pytest.approx(np.mean(vals))
    assert summary.lambda2_abs_min == pytest.approx(min(vals))
    assert summary.lambda2_abs_max == pytest.approx(max(vals))
    assert summary.gap_mean == pytest.approx(np.mean(vals) - bound)
    assert summary.frac_connected == 1.0
    assert summary.frac_bipartite_like == 0.0
    # exactly one of the three trials exceeds the bound
    assert summary.frac_exceeds_bound == pytest.approx(1 / 3)
    # eps=0.05: only the bound+0.2 trial is outside [bound-0.05, bound+0.05]... within (<=) check
    assert summary.frac_within_eps["eps_0.05"] == pytest.approx(2 / 3)


def test_summarize_sweep_groups_by_cell():
    rows = [
        _make_trial(3, 50, 0, alon_boppana_bound(3)),
        _make_trial(3, 50, 1, alon_boppana_bound(3)),
        _make_trial(3, 100, 0, alon_boppana_bound(3)),
    ]
    cells = summarize_sweep(rows)
    assert len(cells) == 2
    cell_by_n = {c.n: c for c in cells}
    assert cell_by_n[50].trials == 2
    assert cell_by_n[100].trials == 1


def test_fit_gap_power_law_recovers_known_exponent():
    d = 4
    bound = alon_boppana_bound(d)
    true_alpha = -0.75
    C = 3.0
    ns = [100, 400, 1600, 6400]
    cells = []
    for n in ns:
        gap = -C * n**true_alpha  # convergence from below, as observed empirically
        cells.append(
            CellSummary(
                d=d,
                n=n,
                bound=bound,
                trials=10,
                lambda2_abs_mean=bound + gap,
                lambda2_abs_std=0.01,
                lambda2_abs_min=bound + gap - 0.01,
                lambda2_abs_max=bound + gap + 0.01,
                gap_mean=gap,
                gap_min=gap - 0.01,
                gap_max=gap + 0.01,
                frac_connected=1.0,
                frac_bipartite_like=0.0,
                frac_exceeds_bound=0.0,
                frac_within_eps={"eps_0.05": 1.0},
                frac_close_to_bound={"eps_0.05": 0.0},
            )
        )
    fit = fit_gap_power_law(cells, d)
    assert fit["alpha"] == pytest.approx(true_alpha, abs=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_fit_gap_power_law_insufficient_points():
    d = 3
    bound = alon_boppana_bound(d)
    cells = [
        CellSummary(
            d=d, n=100, bound=bound, trials=1, lambda2_abs_mean=bound, lambda2_abs_std=0.0,
            lambda2_abs_min=bound, lambda2_abs_max=bound, gap_mean=0.0, gap_min=0.0, gap_max=0.0,
            frac_connected=1.0, frac_bipartite_like=0.0, frac_exceeds_bound=0.0,
            frac_within_eps={}, frac_close_to_bound={},
        )
    ]
    fit = fit_gap_power_law(cells, d)
    assert fit["alpha"] is None


def test_trial_result_and_cell_summary_row_flattening():
    r = _make_trial(3, 100, 0, alon_boppana_bound(3))
    row = trial_result_to_row(r)
    assert row["d"] == 3 and row["n"] == 100

    cells = summarize_sweep([r])
    crow = cell_summary_to_row(cells[0])
    assert "frac_within_eps" not in crow
    assert "frac_close_to_bound" not in crow
    assert any(k.startswith("within_eps_") for k in crow)
    assert any(k.startswith("close_eps_") for k in crow)


@pytest.mark.integration
def test_compare_generators_smoke():
    result = compare_generators(3, 60, trials=10)
    assert result["d"] == 3
    assert result["n"] == 60
    assert result["trials"] == 10
    assert abs(result["diff_in_standard_errors"]) < 10  # sanity, not a strict statistical claim


@pytest.mark.integration
def test_full_small_sweep_end_to_end():
    results = run_sweep(degrees=(3,), n_grid=(50, 100, 200), trials_fn=lambda n: 5)
    cells = summarize_sweep(results)
    assert len(cells) == 3
    fit = fit_gap_power_law(cells, 3)
    # not asserting a specific exponent here (too few/small-n points to be
    # reliable) -- just that the pipeline runs end-to-end and returns a fit
    assert fit["n_points"] <= 3
