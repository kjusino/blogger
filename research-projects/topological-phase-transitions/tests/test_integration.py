import os

from tda_phase_transitions import experiment


def test_run_trial_end_to_end_all_models():
    for model in experiment.MODEL_NAMES:
        result = experiment.run_trial(model, n=30, seed=123, grid_points=50)
        assert result.model == model
        assert result.n == 30
        assert result.theory_threshold > 0
        assert result.percolation_threshold >= 0
        assert result.t_max > 0
        row = result.as_row()
        assert row["percolation_ratio"] == result.percolation_threshold / result.theory_threshold


def test_run_trial_with_curves_shapes_match_grid():
    result, curves = experiment.run_trial(
        "er", n=25, seed=1, grid_points=40, return_curves=True
    )
    assert curves["thresholds"].shape == (40,)
    assert curves["beta0"].shape == (40,)
    assert curves["beta1"].shape == (40,)
    assert curves["giant_frac"].shape == (40,)
    assert curves["susceptibility"].shape == (40,)
    # Giant fraction is non-decreasing and bounded in (0, 1]
    assert (curves["giant_frac"] >= 0).all() and (curves["giant_frac"] <= 1).all()
    assert (curves["giant_frac"][1:] >= curves["giant_frac"][:-1] - 1e-12).all()


def test_run_sweep_and_save_outputs(tmp_path):
    results = experiment.run_sweep(
        models=["er", "rgg"], n_values=[20, 40], trials=3, base_seed=0
    )
    assert len(results) == 2 * 2 * 3

    csv_path = str(tmp_path / "results.csv")
    experiment.save_results_csv(results, csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 1 + len(results)  # header + rows

    summary_path = str(tmp_path / "summary.json")
    summary = experiment.save_summary_json(results, summary_path)
    assert os.path.exists(summary_path)
    assert set(summary.keys()) == {"er", "rgg"}
    for entry in summary.values():
        assert entry["n_trials"] == 6
