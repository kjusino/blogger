import json
import os

import pytest

from ramanujan_spectra import run_experiment


@pytest.mark.integration
def test_main_end_to_end_writes_expected_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_experiment, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(run_experiment, "DEGREES", (3, 4))
    monkeypatch.setattr(run_experiment, "N_GRID", (50, 100, 200))
    monkeypatch.setattr(run_experiment, "GENERATOR_VALIDATION_DEGREES", (3,))
    monkeypatch.setattr(run_experiment, "GENERATOR_VALIDATION_N", (50,))

    from ramanujan_spectra.experiment import trials_for_n

    run_experiment.main()

    expected_total_graphs = len((3, 4)) * sum(trials_for_n(n) for n in (50, 100, 200))

    expected_files = [
        "exact_validation.csv",
        "exact_validation.png",
        "generator_cross_validation.csv",
        "trials.csv",
        "summary.csv",
        "summary.json",
        "lambda2_vs_n.png",
        "gap_convergence_loglog.png",
        "near_ramanujan_fraction.png",
        "distribution_comparison.png",
    ]
    for fname in expected_files:
        path = os.path.join(str(tmp_path), fname)
        assert os.path.exists(path), f"missing expected artifact: {fname}"
        assert os.path.getsize(path) > 0

    with open(os.path.join(str(tmp_path), "summary.json")) as f:
        summary = json.load(f)

    assert summary["degrees"] == [3, 4]
    assert summary["total_graphs"] == expected_total_graphs
    assert summary["exact_validation"]["max_abs_error"] < 1e-6
    assert summary["n_disconnected"] >= 0
