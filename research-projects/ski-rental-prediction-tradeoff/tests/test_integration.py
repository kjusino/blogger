"""End-to-end integration test: runs a small/quick version of the full
experiment pipeline (mirroring `python3 run_experiment.py --quick`) and
checks that it produces sane, finite, correctly-shaped output -- both the
in-memory summary and the actual files on disk (CSVs, summary.json, PNGs)."""

import json
import math
import os
import tempfile

from src.experiment import run_full_experiment


def test_quick_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        figures_dir = os.path.join(tmp, "figures")

        summary = run_full_experiment(results_dir, figures_dir, quick=True, seed=123)

        # --- summary sanity ---
        assert summary["quick_mode"] is True
        assert summary["elapsed_seconds"] > 0
        assert summary["elapsed_seconds"] < 60, "quick mode should run in seconds, not a minute+"

        classical = summary["classical_bound_check"]
        assert 1.0 < classical["robustness_at_lambda0"] < 2.0

        tvb = summary["theory_vs_bruteforce"]
        assert isinstance(tvb["all_within_tolerance"], bool)
        assert math.isfinite(tvb["max_robustness_abs_diff"]["value"])
        assert math.isfinite(tvb["max_consistency_abs_diff"]["value"])

        mc = summary["monte_carlo"]
        for sigma, lam_star in mc["lambda_star_empirical_by_sigma"].items():
            assert 0.0 <= lam_star <= 1.0

        # --- files on disk ---
        expected_result_files = [
            "theory_vs_bruteforce.csv",
            "robustness_vs_b.csv",
            "monte_carlo.csv",
            "lambda_star.csv",
            "summary.json",
        ]
        for fname in expected_result_files:
            path = os.path.join(results_dir, fname)
            assert os.path.isfile(path), f"missing results file: {fname}"
            assert os.path.getsize(path) > 0

        expected_figure_files = [
            "robustness_consistency_tradeoff.png",
            "robustness_vs_b.png",
            "expected_ratio_vs_lambda_by_sigma.png",
            "lambda_star_vs_sigma.png",
        ]
        for fname in expected_figure_files:
            path = os.path.join(figures_dir, fname)
            assert os.path.isfile(path), f"missing figure: {fname}"
            assert os.path.getsize(path) > 1000, f"figure {fname} looks suspiciously small/empty"

        with open(os.path.join(results_dir, "summary.json")) as f:
            reloaded = json.load(f)
        assert reloaded["quick_mode"] is True


def test_quick_pipeline_is_fast():
    import time

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = os.path.join(tmp, "results")
        figures_dir = os.path.join(tmp, "figures")

        start = time.time()
        run_full_experiment(results_dir, figures_dir, quick=True, seed=1)
        elapsed = time.time() - start

        assert elapsed < 30, f"--quick mode took {elapsed:.1f}s, expected a few seconds"
