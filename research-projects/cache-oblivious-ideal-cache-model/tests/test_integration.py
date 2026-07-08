import csv
import json
import os
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_smoke_run_end_to_end_produces_expected_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "run_experiment.py"), "--smoke", "--output-dir", tmp],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr

        raw_csv = os.path.join(tmp, "results", "raw_results.csv")
        summary_json = os.path.join(tmp, "results", "summary.json")
        assert os.path.exists(raw_csv)
        assert os.path.exists(summary_json)

        with open(raw_csv) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        experiments = {r["experiment"] for r in rows}
        assert experiments == {
            "scaling_n", "scaling_B", "scaling_M", "naive_capacity_cliff", "tall_cache_boundary",
        }

        with open(summary_json) as f:
            summary = json.load(f)
        assert summary["config"]["smoke"] is True
        assert "fits" in summary
        assert "naive_capacity_cliff" in summary

        figures_dir = os.path.join(tmp, "figures")
        expected_figures = {
            "scaling_n.png", "scaling_B.png", "scaling_M.png",
            "naive_capacity_cliff.png", "tall_cache_boundary.png", "fit_summary.png",
        }
        assert expected_figures.issubset(set(os.listdir(figures_dir)))
        for fname in expected_figures:
            assert os.path.getsize(os.path.join(figures_dir, fname)) > 0


def test_smoke_run_is_deterministic_given_seed():
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        for tmp in (tmp1, tmp2):
            result = subprocess.run(
                [
                    sys.executable, os.path.join(PROJECT_ROOT, "run_experiment.py"),
                    "--smoke", "--seed", "123", "--output-dir", tmp,
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert result.returncode == 0, result.stderr

        with open(os.path.join(tmp1, "results", "raw_results.csv")) as f:
            rows1 = list(csv.DictReader(f))
        with open(os.path.join(tmp2, "results", "raw_results.csv")) as f:
            rows2 = list(csv.DictReader(f))
        misses1 = [r["misses"] for r in rows1]
        misses2 = [r["misses"] for r in rows2]
        assert misses1 == misses2
