import csv
import json
import os
import tempfile

import numpy as np

from src.experiment import run_sweep, seed_for, summarize


def test_seed_for_is_deterministic_and_distinct_across_inputs():
    a = seed_for(base_seed=0, distance=3, p_index=0)
    b = seed_for(base_seed=0, distance=3, p_index=0)
    c = seed_for(base_seed=0, distance=3, p_index=1)
    d = seed_for(base_seed=0, distance=5, p_index=0)
    assert a == b
    assert a != c
    assert a != d


def test_run_sweep_and_summarize_end_to_end():
    # Small but real Monte Carlo run: enough shots that, at these p values,
    # distance=5 clearly beats distance=3 (deep sub-threshold regime),
    # exercising the full stim -> pymatching -> summary pipeline. p=0.004
    # (rather than something deeper below threshold) keeps the expected
    # logical-error count comfortably nonzero at only 8000 shots.
    distances = [3, 5]
    p_values = np.array([0.004, 0.015])
    results = run_sweep(distances, p_values, shots=8000, base_seed=123)

    assert set(results.keys()) == {3, 5}
    for d in distances:
        assert len(results[d]) == 2
        for r in results[d]:
            assert r.shots == 8000
            assert 0 <= r.logical_errors <= r.shots

    summary = summarize(p_values, results)

    assert summary["distances"] == [3, 5]
    assert len(summary["raw_results"]) == 4
    assert "3" in summary["subthreshold_exponent_fits"]
    assert "5" in summary["subthreshold_exponent_fits"]

    # Sub-threshold signature: at the lowest p, the higher-distance code
    # should have a lower (or equal, given shot noise) logical error rate.
    low_p_rows = [r for r in summary["raw_results"] if r["p"] == p_values[0]]
    rate_by_distance = {r["distance"]: r["logical_error_rate"] for r in low_p_rows}
    assert rate_by_distance[5] <= rate_by_distance[3]


def test_summarize_degrades_gracefully_when_no_logical_errors_observed():
    # p=0 guarantees zero logical errors regardless of shot count, so the
    # sub-threshold power-law fit has no nonzero points to work with.
    # summarize() should record that as an error entry, not raise.
    distances = [3]
    p_values = np.array([0.0, 0.0])
    results = run_sweep(distances, p_values, shots=100, base_seed=1)
    summary = summarize(p_values, results)
    fit = summary["subthreshold_exponent_fits"]["3"]
    assert "error" in fit
    assert fit["predicted_slope"] == 2


def test_summary_is_json_serializable_round_trip():
    distances = [3]
    p_values = np.array([0.005, 0.01])
    results = run_sweep(distances, p_values, shots=2000, base_seed=99)
    summary = summarize(p_values, results)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f)
        with open(path) as f:
            reloaded = json.load(f)
        assert reloaded["distances"] == [3]
        assert len(reloaded["raw_results"]) == 2


def test_csv_export_matches_summary_row_count():
    from run_experiment import write_csv

    distances = [3]
    p_values = np.array([0.005, 0.01, 0.02])
    results = run_sweep(distances, p_values, shots=1000, base_seed=5)
    summary = summarize(p_values, results)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw_results.csv")
        write_csv(summary, path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert set(rows[0].keys()) == {
            "distance", "p", "shots", "logical_errors",
            "logical_error_rate", "ci_lo", "ci_hi",
        }
