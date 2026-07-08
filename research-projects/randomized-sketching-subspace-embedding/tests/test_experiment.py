import json
import os
import subprocess
import sys

from src.experiment import SMOKE_CONFIG, run_all
from src.sketches import SKETCHES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_all_smoke_shapes():
    results = run_all(SMOKE_CONFIG, seed=123)
    n_sketches = len(SKETCHES)

    assert len(results["threshold"]) == n_sketches * len(SMOKE_CONFIG["threshold_multipliers"])
    assert len(results["scaling"]) == n_sketches * len(SMOKE_CONFIG["scaling_k_grid"])
    assert len(results["coherence"]) == 2 * 2 * len(SMOKE_CONFIG["coherence_k_grid"])
    assert len(results["least_squares"]) == n_sketches * len(SMOKE_CONFIG["ls_multipliers"])
    assert len(results["timing_vs_k"]) == n_sketches * len(SMOKE_CONFIG["timing_k_grid"])
    assert len(results["timing_vs_n"]) == n_sketches * len(SMOKE_CONFIG["timing_n_grid"])

    for row in results["threshold"]:
        assert 0.0 <= row["success_rate"] <= 1.0
        assert row["median_eps"] >= 0.0
    for row in results["timing_vs_k"] + results["timing_vs_n"]:
        assert row["time_s"] > 0.0


def test_determinism_same_seed_same_results():
    # timing_vs_k / timing_vs_n are wall-clock measurements and are intentionally
    # excluded: only the statistical sweeps need to be reproducible for a fixed seed.
    r1 = run_all(SMOKE_CONFIG, seed=42)
    r2 = run_all(SMOKE_CONFIG, seed=42)
    assert r1["threshold"] == r2["threshold"]
    assert r1["scaling"] == r2["scaling"]
    assert r1["coherence"] == r2["coherence"]
    assert r1["least_squares"] == r2["least_squares"]


def test_different_seeds_give_different_results():
    r1 = run_all(SMOKE_CONFIG, seed=1)
    r2 = run_all(SMOKE_CONFIG, seed=2)
    assert r1["threshold"] != r2["threshold"]


def test_cli_smoke_run_end_to_end(tmp_path):
    outdir = tmp_path / "results"
    figdir = tmp_path / "figures"
    proc = subprocess.run(
        [sys.executable, "run_experiment.py", "--smoke", "--seed", "7",
         "--outdir", str(outdir), "--figdir", str(figdir)],
        cwd=HERE, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    assert (outdir / "raw_results.csv").exists()
    assert (outdir / "summary.json").exists()
    for i in range(1, 7):
        assert len(list(figdir.glob(f"fig{i}_*.png"))) == 1

    summary = json.loads((outdir / "summary.json").read_text())
    assert summary["k0_threshold"] > 0
    assert set(summary["scaling_fits"].keys()) == set(SKETCHES.keys())
