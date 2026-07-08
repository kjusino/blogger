from pathlib import Path

import numpy as np
import pytest

from src.experiment import (
    aggregate_by_topology_n,
    build_tau_grid,
    compute_heterogeneity_correlation,
    dense_band,
    load_dicts_csv,
    run_network_sweep,
    summarize_network,
    write_dicts_csv,
)


def test_build_tau_grid_brackets_both_predictions():
    grid = build_tau_grid(qmf_tc=0.1, hmf_tc=0.2)
    assert grid.min() < 0.1
    assert grid.max() > 0.2
    assert np.all(np.diff(grid) > 0)  # sorted, unique


def test_dense_band_is_narrower_than_full_grid_and_brackets_predictions():
    band_lo, band_hi = dense_band(qmf_tc=0.1, hmf_tc=0.2)
    grid = build_tau_grid(qmf_tc=0.1, hmf_tc=0.2)
    assert band_lo < 0.1
    assert band_hi > 0.2
    assert band_lo > grid.min()
    assert band_hi < grid.max()


@pytest.mark.parametrize("topology", ["rr", "er", "ba"])
def test_run_network_sweep_end_to_end_small_and_fast(topology):
    # Small network + short sweep: an integration smoke test, not a
    # scientific measurement -- just checks the whole pipeline produces
    # well-formed, self-consistent output quickly.
    result = run_network_sweep(
        topology, n=60, mean_degree=6.0, delta=0.2,
        n_steps=400, burn_in=150, n_repeats=2, seed=1,
    )
    assert result["topology"] == topology
    assert result["n"] == 60
    assert len(result["sweep_rows"]) == len(result["tau_grid"])
    assert result["tau_c_empirical"] > 0
    assert not np.isnan(result["tau_c_sem"])

    for row in result["sweep_rows"]:
        assert 0.0 <= row["mean_rho_mean"] <= 1.0
        assert row["susceptibility_mean"] >= 0 or np.isnan(row["susceptibility_mean"])

    summary = summarize_network(result)
    assert summary["qmf_threshold"] > 0
    assert summary["hmf_threshold"] > 0
    assert summary["heterogeneity_ratio"] >= 1.0 - 1e-6


def test_heterogeneity_correlation_on_synthetic_perfectly_ranked_data():
    summaries = [
        {"heterogeneity_ratio": 1.0, "gap_hmf_minus_qmf": 0.01},
        {"heterogeneity_ratio": 1.5, "gap_hmf_minus_qmf": 0.05},
        {"heterogeneity_ratio": 2.0, "gap_hmf_minus_qmf": 0.10},
        {"heterogeneity_ratio": 3.0, "gap_hmf_minus_qmf": 0.20},
    ]
    result = compute_heterogeneity_correlation(summaries)
    assert result["n_points"] == 4
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["spearman_p"] < 0.1


def test_heterogeneity_correlation_ignores_nan_gaps():
    summaries = [
        {"heterogeneity_ratio": 1.0, "gap_hmf_minus_qmf": 0.01},
        {"heterogeneity_ratio": 2.0, "gap_hmf_minus_qmf": float("nan")},
        {"heterogeneity_ratio": 3.0, "gap_hmf_minus_qmf": 0.2},
    ]
    result = compute_heterogeneity_correlation(summaries)
    assert result["n_points"] == 2


def test_aggregate_by_topology_n_averages_across_realizations():
    summaries = [
        {"topology": "rr", "n": 100, "mean_degree": 6.0, "mean_sq_degree": 36.0,
         "heterogeneity_ratio": 1.0, "lambda_max": 6.0, "qmf_threshold": 0.1667,
         "hmf_threshold": 0.1667, "tau_c_empirical": 0.16, "eps_qmf": 0.04, "eps_hmf": 0.04,
         "gap_hmf_minus_qmf": 0.0, "realization": 0},
        {"topology": "rr", "n": 100, "mean_degree": 6.0, "mean_sq_degree": 36.0,
         "heterogeneity_ratio": 1.0, "lambda_max": 6.0, "qmf_threshold": 0.1667,
         "hmf_threshold": 0.1667, "tau_c_empirical": 0.18, "eps_qmf": 0.08, "eps_hmf": 0.08,
         "gap_hmf_minus_qmf": 0.0, "realization": 1},
    ]
    aggregated = aggregate_by_topology_n(summaries, topologies=["rr"], ns=[100])
    assert len(aggregated) == 1
    row = aggregated[0]
    assert row["n_realizations"] == 2
    assert row["tau_c_empirical"] == pytest.approx(0.17)
    assert not np.isnan(row["tau_c_sem"])


def test_aggregate_by_topology_n_skips_missing_combinations():
    summaries = [
        {"topology": "rr", "n": 100, "mean_degree": 6.0, "mean_sq_degree": 36.0,
         "heterogeneity_ratio": 1.0, "lambda_max": 6.0, "qmf_threshold": 0.1667,
         "hmf_threshold": 0.1667, "tau_c_empirical": 0.16, "eps_qmf": 0.04, "eps_hmf": 0.04,
         "gap_hmf_minus_qmf": 0.0, "realization": 0},
    ]
    aggregated = aggregate_by_topology_n(summaries, topologies=["rr", "ba"], ns=[100, 200])
    assert len(aggregated) == 1


def test_csv_roundtrip(tmp_path: Path):
    rows = [
        {"topology": "rr", "n": 100, "tau": 0.123, "value": 4.5},
        {"topology": "ba", "n": 200, "tau": 0.456, "value": 7.8},
    ]
    path = tmp_path / "test.csv"
    write_dicts_csv(rows, path)
    loaded = load_dicts_csv(path)
    assert loaded[0]["topology"] == "rr"
    assert loaded[0]["n"] == 100
    assert loaded[1]["tau"] == pytest.approx(0.456)
    assert loaded[1]["value"] == pytest.approx(7.8)
