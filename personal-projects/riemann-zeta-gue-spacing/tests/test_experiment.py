import numpy as np
import pytest

from src.experiment import analyze_window, nested_subsample_metrics, loglog_fit


def test_analyze_window_end_to_end_small():
    # Small enough (15 zeros) to run in well under a second, but big
    # enough to exercise every metric in the pipeline.
    result = analyze_window(1, 15, u_max=2.0, n_bins=8, label="tiny")
    assert result["n_spacings"] == 14
    assert result["mean_spacing"] > 0
    assert 0.0 <= result["ks_gue"] <= 1.0
    assert 0.0 <= result["p_gue"] <= 1.0
    assert 0.0 <= result["ks_poisson"] <= 1.0
    assert result["pc_err_montgomery"] >= 0.0
    assert 0.0 <= result["repulsion_fraction"] <= 1.0
    assert result["unfolded"].shape == (15,)
    assert result["label"] == "tiny"


def test_analyze_window_default_label():
    result = analyze_window(1, 10, u_max=2.0, n_bins=5)
    assert result["label"] == "n=1"


def test_nested_subsample_metrics_uses_growing_prefixes():
    base = analyze_window(1, 21, u_max=2.0, n_bins=8, label="base")
    subs = nested_subsample_metrics(base, [5, 10, 20], u_max=2.0, n_bins=8)
    assert [r["n_spacings"] for r in subs] == [5, 10, 20]
    # The N=20 prefix's mean spacing must equal the full window's, since
    # it *is* the full window (21 zeros -> 20 spacings).
    assert subs[-1]["mean_spacing"] == pytest.approx(base["mean_spacing"])


def test_nested_subsample_metrics_rejects_oversized_n():
    base = analyze_window(1, 10, u_max=2.0, n_bins=5, label="base")
    with pytest.raises(ValueError):
        nested_subsample_metrics(base, [50], u_max=2.0, n_bins=5)


def test_loglog_fit_recovers_known_power_law():
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y = 3.0 * x ** -0.5
    fit = loglog_fit(x, y)
    assert fit["slope"] == pytest.approx(-0.5, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)
