import os

import numpy as np

from src.experiment import (
    empirical_threshold,
    run_grid_sweep,
    theory_rmse,
    transition_width,
)
from src.plots import plot_heatmaps, plot_threshold_curves, plot_transition_width


def test_run_grid_sweep_produces_expected_number_of_points():
    n_values = [40, 60]
    delta_grid = np.linspace(0.2, 0.8, 4)
    rho_grid = np.linspace(0.1, 0.4, 3)
    result = run_grid_sweep(n_values, delta_grid, rho_grid, trials=3, seed=42)
    assert len(result.points) == len(n_values) * len(delta_grid) * len(rho_grid)
    for p in result.points:
        assert 0.0 <= p.success_rate <= 1.0
        assert p.trials == 3


def test_success_matrix_shape_and_values():
    n_values = [40]
    delta_grid = np.linspace(0.2, 0.8, 4)
    rho_grid = np.linspace(0.1, 0.3, 2)
    result = run_grid_sweep(n_values, delta_grid, rho_grid, trials=2, seed=1)
    mat = result.success_matrix(40)
    assert mat.shape == (len(rho_grid), len(delta_grid))
    assert not np.isnan(mat).any()


def test_grid_sweep_recovers_qualitative_phase_transition():
    # At small rho, a low-delta grid point should fail more often than a
    # high-delta grid point, i.e. success probability should be
    # non-decreasing (on average) as delta grows, reproducing the expected
    # qualitative shape of the phase transition on a coarse grid.
    n = 60
    rho = 0.1
    delta_grid = np.array([0.1, 0.9])
    result = run_grid_sweep([n], delta_grid, np.array([rho]), trials=20, seed=7)
    mat = result.success_matrix(n)
    low_delta_rate, high_delta_rate = mat[0, 0], mat[0, 1]
    assert high_delta_rate >= low_delta_rate


def test_empirical_threshold_linear_interpolation():
    delta_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    rates = np.array([0.0, 0.0, 0.4, 0.8, 1.0])
    thresh = empirical_threshold(delta_grid, rates)
    # Crosses 0.5 between delta=0.5 (rate 0.4) and delta=0.75 (rate 0.8).
    assert 0.5 < thresh < 0.75


def test_empirical_threshold_none_when_never_crosses():
    delta_grid = np.array([0.0, 0.5, 1.0])
    rates = np.array([0.0, 0.1, 0.2])
    assert empirical_threshold(delta_grid, rates) is None


def test_empirical_threshold_immediate_success():
    delta_grid = np.array([0.0, 0.5, 1.0])
    rates = np.array([0.9, 0.95, 1.0])
    assert empirical_threshold(delta_grid, rates) == 0.0


def test_transition_width_narrower_for_steeper_curve():
    delta_grid = np.linspace(0, 1, 11)
    steep = np.where(delta_grid < 0.5, 0.0, 1.0)
    shallow = np.clip((delta_grid - 0.1) / 0.8, 0, 1)
    w_steep = transition_width(delta_grid, steep)
    w_shallow = transition_width(delta_grid, shallow)
    assert w_steep < w_shallow


def test_theory_rmse_is_small_for_large_n(tmp_path):
    # A modest but not tiny n, several trials: empirical thresholds should
    # land reasonably close to the theoretical curve.
    n = 120
    delta_grid = np.linspace(0.1, 0.9, 9)
    rho_grid = np.array([0.1, 0.2, 0.3])
    result = run_grid_sweep([n], delta_grid, rho_grid, trials=15, seed=123)
    rmse = theory_rmse(result, n)
    assert rmse == rmse  # not NaN
    assert rmse < 0.15


def test_plotting_pipeline_writes_files(tmp_path):
    n_values = [30, 50]
    delta_grid = np.linspace(0.2, 0.8, 4)
    rho_grid = np.linspace(0.1, 0.3, 3)
    result = run_grid_sweep(n_values, delta_grid, rho_grid, trials=3, seed=99)

    heatmap_path = tmp_path / "heatmaps.png"
    threshold_path = tmp_path / "thresholds.png"
    width_path = tmp_path / "width.png"

    plot_heatmaps(result, str(heatmap_path))
    plot_threshold_curves(result, str(threshold_path))

    widths = []
    for n in n_values:
        mat = result.success_matrix(n)
        row_widths = [transition_width(delta_grid, mat[i]) for i in range(len(rho_grid))]
        row_widths = [w for w in row_widths if w is not None]
        widths.append(np.mean(row_widths) if row_widths else float("nan"))
    plot_transition_width(n_values, widths, str(width_path))

    for path in (heatmap_path, threshold_path, width_path):
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
