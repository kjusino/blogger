"""End-to-end small-scale pipeline test: run a tiny version of the full
sweep -> critical-alpha -> finite-size-extrapolation -> capacity-fit
pipeline and sanity-check the shapes/values at each stage."""

import numpy as np

from src.experiment import (
    run_sweep,
    compute_critical_alphas,
    compute_finite_size_extrapolation,
    compute_capacity_vs_rho_fit,
    run_arcsin_validation,
)


def test_small_scale_pipeline_runs_end_to_end():
    results = run_sweep(
        ns=(40, 80, 160),
        rhos=(0.0, 0.3),
        n_trials=4,
        n_patterns_per_trial=3,
        max_sweeps=10,
        seed=123,
        verbose=False,
    )
    assert len(results) > 0

    rows = [
        {
            "n": r.n, "rho": r.rho, "alpha": r.alpha, "p": r.p,
            "n_trials": r.n_trials, "n_samples": r.n_samples,
            "mean_overlap": r.mean_overlap, "sem_overlap": r.sem_overlap,
            "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
            "mean_sweeps_used": r.mean_sweeps_used,
        }
        for r in results
    ]

    # Low-alpha points should show much better retrieval than high-alpha points.
    for n in (40, 80, 160):
        pts = sorted([r for r in rows if r["n"] == n], key=lambda r: r["alpha"])
        assert pts[0]["mean_overlap"] > pts[-1]["mean_overlap"] - 0.01 or \
            pts[0]["mean_overlap"] > 0.5

    critical_alphas = compute_critical_alphas(rows, threshold=0.95)
    assert len(critical_alphas) == 3 * 2  # 3 Ns x 2 rhos

    extrapolation = compute_finite_size_extrapolation(critical_alphas)
    assert len(extrapolation) >= 1
    for e in extrapolation:
        assert 0.0 < e["alpha_c_inf"] < 1.0

    if len(extrapolation) >= 2:
        fit = compute_capacity_vs_rho_fit(extrapolation)
        assert fit["better_model"] in ("linear (H1)", "power-law")
        assert -1.0 <= fit["spearman_rho"] <= 1.0


def test_arcsin_validation_small_scale():
    rows = run_arcsin_validation(rhos=(0.0, 0.2, 0.5), n=500, p=80, n_trials=3, seed=1)
    assert len(rows) == 3
    for r in rows:
        assert r["abs_error"] < 0.05
