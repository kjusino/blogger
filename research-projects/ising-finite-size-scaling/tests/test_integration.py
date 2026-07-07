import numpy as np

from src.experiment import (
    run_fss_grid,
    analyze_scaling,
    onsager_validation,
    autocorrelation_comparison,
)
from src import theory


def test_fss_grid_runs_end_to_end_and_shapes_are_consistent():
    L_values = [8, 16]
    T_grid = np.linspace(2.0, 2.5, 5)
    rows, by_L = run_fss_grid(
        L_values, T_grid, n_equil=20, n_sample=20, sample_interval=1, n_seeds=2
    )
    assert len(rows) == len(L_values) * len(T_grid)
    for row in rows:
        assert row["L"] in L_values
        assert -2.0 <= row["e_mean"] <= 0.0
        assert 0.0 <= row["m_mean"] <= 1.0
        assert row["chi_mean"] >= 0.0
        assert row["c_mean"] >= 0.0

    for L in L_values:
        assert by_L[L]["T"].shape == (5,)
        assert by_L[L]["m"].shape == (5,)


def test_analyze_scaling_produces_finite_estimates():
    L_values = [8, 16, 32]
    T_grid = np.linspace(2.0, 2.5, 8)
    rows, by_L = run_fss_grid(
        L_values, T_grid, n_equil=30, n_sample=30, sample_interval=1, n_seeds=2
    )
    result = analyze_scaling(by_L, L_values)
    assert result["raw_collapse_rmse"] is not None
    assert result["rescaled_magnetization_collapse_rmse"] is not None
    for L in L_values:
        assert result["x_by_L"][L].shape == (8,)
        assert result["y_by_L"][L].shape == (8,)


def test_onsager_validation_runs_and_reports_small_errors_at_large_L():
    rows = onsager_validation(
        L_large=32,
        T_values=[1.5, 3.0],
        n_equil=100,
        n_sample=100,
        sample_interval=1,
        n_seeds=2,
    )
    assert len(rows) == 2
    for row in rows:
        # Away from Tc, a modest L=32 lattice should already track the exact
        # infinite-lattice energy reasonably well.
        assert row["energy_abs_error"] < 0.3


def test_autocorrelation_comparison_runs_and_returns_positive_taus():
    rows = autocorrelation_comparison(L=8, T_values=[theory.T_C], n_equil=20, n_sample=50)
    assert len(rows) == 1
    row = rows[0]
    assert row["metropolis_tau_sweeps"] > 0
    assert row["wolff_tau_steps"] > 0
    assert row["mean_wolff_cluster_size"] > 0
