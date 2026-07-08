#!/usr/bin/env python3
"""Single reproducible entry point for the Hopfield correlated-capacity
experiment. Runs the full (N, rho, alpha) sweep, the arcsin-law generator
validation, the critical-alpha / finite-size-scaling / capacity-vs-rho
post-processing, writes all raw results to results/*.csv, and regenerates
figures/*.png strictly from those CSVs.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.experiment import (
    DEFAULT_NS,
    DEFAULT_RHOS,
    CLASSICAL_ALPHA_C,
    run_sweep,
    write_sweep_csv,
    load_sweep_csv,
    compute_critical_alphas,
    compute_finite_size_extrapolation,
    compute_capacity_vs_rho_fit,
    run_arcsin_validation,
    write_dicts_csv,
)
from src.plotting import (
    plot_overlap_vs_alpha,
    plot_finite_size_scaling,
    plot_phase_diagram,
    plot_arcsin_validation,
)

SEED = 12345
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    t0 = time.time()
    print(f"=== Hopfield correlated-capacity experiment (seed={SEED}) ===")

    print("\n[1/5] Running (N, rho, alpha) retrieval sweep...")
    sweep_results = run_sweep(
        ns=DEFAULT_NS,
        rhos=DEFAULT_RHOS,
        n_trials=20,
        n_patterns_per_trial=5,
        flip_frac=0.05,
        max_sweeps=20,
        seed=SEED,
        verbose=True,
    )
    write_sweep_csv(sweep_results, RESULTS_DIR / "raw_overlap_sweep.csv")
    print(f"  wrote {len(sweep_results)} rows -> results/raw_overlap_sweep.csv "
          f"({time.time() - t0:.1f}s elapsed)")

    print("\n[2/5] Running arcsin-law pattern-generator validation...")
    arcsin_rows = run_arcsin_validation(seed=SEED + 1)
    write_dicts_csv(arcsin_rows, RESULTS_DIR / "arcsin_law_validation.csv")
    print(f"  wrote {len(arcsin_rows)} rows -> results/arcsin_law_validation.csv")

    print("\n[3/5] Localizing critical alpha per (N, rho)...")
    rows = load_sweep_csv(RESULTS_DIR / "raw_overlap_sweep.csv")
    critical_alphas = compute_critical_alphas(rows, threshold=0.95)
    write_dicts_csv(critical_alphas, RESULTS_DIR / "critical_alpha.csv")
    print(f"  wrote {len(critical_alphas)} rows -> results/critical_alpha.csv")

    print("\n[4/5] Finite-size scaling extrapolation (alpha_c(N) -> alpha_c(inf))...")
    extrapolation = compute_finite_size_extrapolation(critical_alphas)
    write_dicts_csv(extrapolation, RESULTS_DIR / "finite_size_extrapolation.csv")
    print(f"  wrote {len(extrapolation)} rows -> results/finite_size_extrapolation.csv")
    for e in extrapolation:
        print(f"    rho={e['rho']}: alpha_c(inf) = {e['alpha_c_inf']:.4f} "
              f"+/- {e['alpha_c_inf_stderr']:.4f}")

    print("\n[5/5] Fitting linear (H1) vs power-law model for alpha_c(rho)...")
    fit = compute_capacity_vs_rho_fit(extrapolation)
    write_dicts_csv([fit], RESULTS_DIR / "capacity_vs_rho_fit.csv")
    print(f"  alpha_c(0) = {fit['alpha0']:.4f} (classical literature: {CLASSICAL_ALPHA_C})")
    print(f"  linear RMSE={fit['linear_rmse']:.5f} R2={fit['linear_r2']:.4f}")
    print(f"  power  RMSE={fit['power_rmse']:.5f} R2={fit['power_r2']:.4f} k={fit['power_k']:.3f}")
    print(f"  better model: {fit['better_model']}")
    print(f"  Spearman rho={fit['spearman_rho']:.4f} p={fit['spearman_p']:.5g}")

    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_overlap_vs_alpha(rows, FIGURES_DIR / "overlap_vs_alpha.png")
    plot_finite_size_scaling(critical_alphas, extrapolation, FIGURES_DIR / "finite_size_scaling.png")
    plot_phase_diagram(extrapolation, fit, FIGURES_DIR / "phase_diagram.png", CLASSICAL_ALPHA_C)
    plot_arcsin_validation(arcsin_rows, FIGURES_DIR / "arcsin_law_validation.png")
    print("  wrote figures/overlap_vs_alpha.png")
    print("  wrote figures/finite_size_scaling.png")
    print("  wrote figures/phase_diagram.png")
    print("  wrote figures/arcsin_law_validation.png")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
