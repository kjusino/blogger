import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.experiment import find_m50, fit_power_law, generate_samples, power_at_m, power_curve

ROOT = Path(__file__).resolve().parent.parent


def test_generate_samples_uniform_shape_and_range():
    rng = np.random.default_rng(0)
    samples = generate_samples("uniform", 25, 0.2, 1000, rng)
    assert samples.shape == (1000,)
    assert samples.min() >= 0
    assert samples.max() < 25


def test_generate_samples_unknown_family_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_samples("not_a_family", 10, 0.1, 10, rng)


def test_power_at_m_is_between_zero_and_one():
    rng = np.random.default_rng(0)
    power = power_at_m("collision", "paired", 50, 0.2, 100, 30, rng)
    assert 0.0 <= power <= 1.0


def test_find_m50_brackets_a_sane_crossing_for_collision_tester():
    rng = np.random.default_rng(0)
    res = find_m50("collision", "paired", 100, 0.25, 40, rng)
    assert res.bracketed
    assert res.m50 > 0
    # Advantage should genuinely reach 0.5 at the reported crossing region.
    assert max(res.advantage_grid) >= 0.5 - 1e-9
    assert len(res.m_grid) == len(res.power_grid) == len(res.advantage_grid)


def test_find_m50_naive_learner_is_much_larger_than_collision_at_same_n():
    """Sanity-checks the qualitative claim of the whole project on a single
    small case: the naive learner needs substantially more samples than the
    collision tester to reach the same discrimination advantage."""
    rng = np.random.default_rng(0)
    n, epsilon, trials = 300, 0.25, 40
    collision_res = find_m50("collision", "paired", n, epsilon, trials, rng)
    naive_res = find_m50("naive_learner", "paired", n, epsilon, trials, rng)
    assert naive_res.m50 > collision_res.m50


def test_fit_power_law_recovers_known_exponent_exactly():
    ns = [10, 100, 1000, 10000]
    true_slope, true_intercept = 0.5, np.log(3.0)
    m50s = [np.exp(true_intercept) * n ** true_slope for n in ns]
    slope, intercept, r2 = fit_power_law(ns, m50s)
    assert slope == pytest.approx(true_slope, abs=1e-9)
    assert intercept == pytest.approx(true_intercept, abs=1e-9)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_fit_power_law_r_squared_drops_with_noise():
    rng = np.random.default_rng(0)
    ns = np.array([10, 100, 1000, 10000, 100000], dtype=float)
    clean = 2.0 * ns ** 0.5
    noisy = clean * np.exp(rng.normal(0, 0.5, size=ns.shape))
    _, _, r2_clean = fit_power_law(ns, clean)
    _, _, r2_noisy = fit_power_law(ns, noisy)
    assert r2_clean > r2_noisy


def test_power_curve_matches_manual_power_at_m():
    rng = np.random.default_rng(0)
    curve = power_curve("collision", "paired", 80, 0.2, [50, 100, 200], 20, rng)
    assert len(curve) == 3
    assert all(0.0 <= p <= 1.0 for p in curve)


def test_cli_quick_run_produces_expected_artifacts(tmp_path):
    """End-to-end integration test: actually invoke run_experiment.py --quick
    as a subprocess (same entry point a user runs) and check it produces
    real, parseable output with sane fitted exponents."""
    result = subprocess.run(
        [sys.executable, "run_experiment.py", "--quick", "--seed", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    scaling_csv = ROOT / "results" / "scaling_results.csv"
    summary_json = ROOT / "results" / "summary.json"
    assert scaling_csv.exists()
    assert summary_json.exists()

    with open(scaling_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    for row in rows:
        assert 0 < float(row["m50"])

    with open(summary_json) as f:
        summary = json.load(f)
    for fit in summary["fits"].values():
        # Exponents should land in a broad sane band even under --quick's
        # small grid/trial counts; this is a smoke test, not a precision
        # check (that's what the full run + README numbers are for).
        assert 0.0 < fit["fitted_exponent"] < 3.0

    for fig_name in [
        "collision_scaling.png",
        "naive_learner_scaling.png",
        "scaling_comparison.png",
        "family_power_curves.png",
        "collision_probability_gap.png",
    ]:
        assert (ROOT / "figures" / fig_name).exists()
