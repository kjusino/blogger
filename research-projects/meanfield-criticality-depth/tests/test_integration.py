import os
import sys
import json
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment import run_grid_sweep, write_csv, write_json, summarize
import plotting


def test_full_pipeline_tiny_grid_writes_all_artifacts():
    """End-to-end smoke test: run a 2x2 grid through the exact same code
    path as run_experiment.py (sweep -> CSV/JSON -> all four figures) and
    check every artifact lands on disk and is well-formed. Doesn't assert
    on scientific conclusions (that's covered elsewhere) -- just that the
    pipeline itself doesn't break."""
    sigma_w2_grid = np.array([0.8, 3.2])
    sigma_b2_grid = np.array([0.05, 0.2])

    rows = run_grid_sweep(sigma_w2_grid, sigma_b2_grid, seed=0, log=lambda *a, **k: None)
    assert len(rows) == 4

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "grid_results.csv")
        write_csv(rows, csv_path)
        assert os.path.getsize(csv_path) > 0
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        assert "chi1_theory" in header
        assert "max_trainable_depth" in header

        summary = summarize(rows)
        json_path = os.path.join(tmp, "summary.json")
        write_json(summary, json_path)
        with open(json_path) as f:
            reloaded = json.load(f)
        assert reloaded["n_grid_points"] == 4

        plotting.plot_correlation_maps(os.path.join(tmp, "correlation_maps.png"))
        plotting.plot_signal_propagation(os.path.join(tmp, "signal_propagation_examples.png"))
        plotting.plot_phase_diagram(rows, sigma_w2_grid, sigma_b2_grid, os.path.join(tmp, "phase_diagram.png"))
        plotting.plot_depth_vs_xi(rows, os.path.join(tmp, "depth_vs_correlation_length.png"))

        for fname in [
            "correlation_maps.png",
            "signal_propagation_examples.png",
            "phase_diagram.png",
            "depth_vs_correlation_length.png",
        ]:
            path = os.path.join(tmp, fname)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # not an empty/broken image
