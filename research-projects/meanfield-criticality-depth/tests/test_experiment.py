import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment import (
    empirical_correlation_chi,
    empirical_gradient_chi,
    empirical_max_trainable_depth,
    fit_log_decay_rate,
    run_grid_sweep,
    summarize,
)
from meanfield import fixed_point_q, tanh


def test_fit_log_decay_rate_recovers_known_exponential():
    rate_true = 0.05
    values = np.exp(-rate_true * np.arange(200))
    rate_hat, chi_hat = fit_log_decay_rate(values)
    assert rate_hat == pytest.approx(rate_true, rel=0.05)
    assert chi_hat == pytest.approx(np.exp(-rate_true), rel=0.05)


def test_fit_log_decay_rate_handles_all_below_floor():
    rate_hat, chi_hat = fit_log_decay_rate(np.full(20, 1e-10), floor=1e-7)
    assert np.isnan(rate_hat) and np.isnan(chi_hat)


def test_empirical_correlation_chi_matches_theory_reasonably():
    sigma_w2, sigma_b2 = 0.8, 0.05
    q_star = fixed_point_q(sigma_w2, sigma_b2, tanh)
    chi_hat, c_inf, curve = empirical_correlation_chi(sigma_w2, sigma_b2, q_star, depth=50, width=400)
    assert chi_hat == pytest.approx(0.6468, rel=0.15)
    assert c_inf == pytest.approx(1.0, abs=1e-3)
    assert len(curve) == 50


def test_empirical_gradient_chi_matches_theory_reasonably():
    sigma_w2, sigma_b2 = 3.2, 0.1
    chi_hat, curve = empirical_gradient_chi(sigma_w2, sigma_b2, depth=40, width=150, n_draws=3)
    assert chi_hat == pytest.approx(1.2372, rel=0.1)
    assert len(curve) == 40


def test_empirical_max_trainable_depth_monotonic_staircase_semantics():
    # A very ordered, high-depth setting should fail to clear the shallowest
    # depths handled by the staircase eventually; a shallow-only staircase
    # around an easy regime should succeed everywhere.
    depth_easy = empirical_max_trainable_depth(
        sigma_w2=1.9861, sigma_b2=0.1, depths=[2, 4, 8], n_seeds=1
    )
    assert depth_easy == 8  # both shallow depths should train fine

    depth_hard = empirical_max_trainable_depth(
        sigma_w2=0.3, sigma_b2=0.01, depths=[2, 100], n_seeds=1
    )
    assert depth_hard in (0, 2)  # can't have "trained" at depth 100 without depth 2


def test_run_grid_sweep_and_summarize_end_to_end_tiny_grid():
    rows = run_grid_sweep(np.array([0.8, 3.2]), np.array([0.05]), seed=0, log=lambda *a, **k: None)
    assert len(rows) == 2
    for r in rows:
        assert r["phase"] in ("ordered", "chaotic")
        assert r["max_trainable_depth"] >= 0

    summary = summarize(rows)
    assert summary["n_grid_points"] == 2
    assert "forward_correlation_chi1" in summary
    assert "backprop_gradient_chi1" in summary
    assert "trainable_depth_vs_correlation_length" in summary
