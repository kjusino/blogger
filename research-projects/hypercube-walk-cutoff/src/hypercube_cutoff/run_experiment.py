"""End-to-end driver: run all sweeps, write results/, generate all plots.

Usage:
    python -m hypercube_cutoff.run_experiment
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import experiment as exp
from . import plots as plotmod

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

LUMPING_N_VALUES = [8, 10, 12]
LUMPING_TRIALS = 20000
LUMPING_SEED = 0

SCALING_N_VALUES = [50, 100, 200, 400, 800, 1600, 3200]
SCALING_C_VALUES = np.linspace(-4, 6, 41)
PLOT_N_SUBSET = [100, 400, 1600, 3200]

MC_N = 30
MC_C_VALUES = np.linspace(-3, 5, 13)
MC_TRIALS = 20000
MC_SEED = 42


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("1/3 Validating the Hamming-weight lumping (small n: brute force vs. exact vs. Monte Carlo)...")
    lumping_rows = exp.validate_lumping(LUMPING_N_VALUES, num_trials=LUMPING_TRIALS, seed=LUMPING_SEED)
    _write_csv(os.path.join(RESULTS_DIR, "lumping_validation.csv"), lumping_rows)
    max_lumping_diff = max(r["lumped_vs_bruteforce_abs_diff"] for r in lumping_rows)
    mc_agreement_rate = sum(r["mc_within_ci"] for r in lumping_rows) / len(lumping_rows)
    print(f"   max |lumped - brute force| = {max_lumping_diff:.2e}; "
          f"MC-within-CI rate = {mc_agreement_rate:.2%}")

    print("2/3 Cutoff-location and data-collapse sweep across n "
          f"(n={SCALING_N_VALUES})...")
    curve_rows, summary_rows = exp.cutoff_scaling_sweep(SCALING_N_VALUES, SCALING_C_VALUES)
    _write_csv(os.path.join(RESULTS_DIR, "cutoff_scaling_curves.csv"), curve_rows)
    _write_csv(os.path.join(RESULTS_DIR, "cutoff_scaling_summary.csv"), summary_rows)
    for r in summary_rows:
        print(f"   n={r['n']:5d}  t_half={r['t_half_empirical']:9.1f}  "
              f"t*={r['t_star_theory']:9.1f}  rel_err={r['rel_err_half_vs_cutoff_time']:.4f}  "
              f"window/n={r['window_over_n']:.3f}  self_collapse={r['self_collapse_error']:.4f}  "
              f"chi_sq_bound_gap={r['chi_square_bound_gap']:.4f}")

    print(f"3/3 Monte Carlo validation at n={MC_N}...")
    mc_rows = exp.monte_carlo_validation(MC_N, MC_C_VALUES, MC_TRIALS, seed=MC_SEED)
    _write_csv(os.path.join(RESULTS_DIR, "mc_validation.csv"), mc_rows)
    bv_agree = sum(r["bitvector_within_ci"] for r in mc_rows) / len(mc_rows)
    w_agree = sum(r["weight_within_ci"] for r in mc_rows) / len(mc_rows)
    print(f"   bitvector-MC-within-CI rate = {bv_agree:.2%}; weight-MC-within-CI rate = {w_agree:.2%}")

    print("Generating plots...")
    plotmod.plot_tv_curves_vs_t(curve_rows, PLOT_N_SUBSET, os.path.join(RESULTS_DIR, "tv_curves_vs_t.png"))
    plotmod.plot_data_collapse(curve_rows, PLOT_N_SUBSET, os.path.join(RESULTS_DIR, "data_collapse.png"))
    plotmod.plot_cutoff_time_scaling(summary_rows, os.path.join(RESULTS_DIR, "cutoff_time_scaling.png"))
    plotmod.plot_window_scaling(summary_rows, os.path.join(RESULTS_DIR, "window_scaling.png"))
    plotmod.plot_collapse_errors(summary_rows, os.path.join(RESULTS_DIR, "collapse_errors.png"))
    plotmod.plot_mc_validation(mc_rows, os.path.join(RESULTS_DIR, "mc_validation.png"))
    plotmod.plot_lumping_check(lumping_rows, os.path.join(RESULTS_DIR, "lumping_check.png"))

    n_vals = np.array([r["n"] for r in summary_rows], dtype=float)
    window = np.array([r["window"] for r in summary_rows], dtype=float)
    window_slope, _ = np.polyfit(np.log(n_vals), np.log(window), 1)

    summary = {
        "lumping_validation": {
            "n_values": LUMPING_N_VALUES,
            "max_abs_diff_lumped_vs_bruteforce": max_lumping_diff,
            "mc_within_ci_rate": mc_agreement_rate,
        },
        "cutoff_scaling": {
            "n_values": SCALING_N_VALUES,
            "rel_err_by_n": {r["n"]: r["rel_err_half_vs_cutoff_time"] for r in summary_rows},
            "window_over_n_by_n": {r["n"]: r["window_over_n"] for r in summary_rows},
            "self_collapse_error_by_n": {r["n"]: r["self_collapse_error"] for r in summary_rows},
            "chi_square_bound_gap_by_n": {r["n"]: r["chi_square_bound_gap"] for r in summary_rows},
            "window_scaling_fit_slope": float(window_slope),
        },
        "mc_validation": {
            "n": MC_N,
            "num_trials": MC_TRIALS,
            "bitvector_within_ci_rate": bv_agree,
            "weight_within_ci_rate": w_agree,
        },
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
