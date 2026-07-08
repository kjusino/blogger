#!/usr/bin/env python3
"""Run the full compressed-sensing phase-transition sweep and write results
(CSV + JSON summary) to results/ and figures to figures/.

Usage:
    python run_experiment.py            # full sweep (several minutes)
    python run_experiment.py --quick    # small sweep, for a fast smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from src.experiment import empirical_threshold, run_grid_sweep, theory_rmse, transition_width
from src.plots import plot_heatmaps, plot_threshold_curves, plot_transition_width
from src.theory import phase_transition_delta

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Small grid / few trials, for a fast smoke test.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.quick:
        n_values = [40, 80]
        delta_grid = np.linspace(0.1, 0.9, 6)
        rho_grid = np.linspace(0.05, 0.45, 5)
        trials = 8
    else:
        n_values = [60, 120, 240]
        delta_grid = np.linspace(0.05, 0.95, 13)
        rho_grid = np.linspace(0.05, 0.55, 13)
        trials = 20

    print(f"Grid: n={n_values}, delta_grid={len(delta_grid)} pts, "
          f"rho_grid={len(rho_grid)} pts, trials={trials} "
          f"=> {len(n_values) * len(delta_grid) * len(rho_grid)} grid points, "
          f"{len(n_values) * len(delta_grid) * len(rho_grid) * trials} LP solves")

    t0 = time.time()
    result = run_grid_sweep(n_values, delta_grid, rho_grid, trials=trials, seed=2024)
    elapsed = time.time() - t0
    print(f"Sweep finished in {elapsed:.1f}s")

    # --- CSV of every grid point ---
    csv_path = os.path.join(RESULTS_DIR, "grid_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "delta", "rho", "m", "k", "success_rate", "trials"])
        for p in result.points:
            writer.writerow([p.n, p.delta, p.rho, p.m, p.k, p.success_rate, p.trials])
    print(f"Wrote {csv_path}")

    # --- per-n threshold comparison + summary stats ---
    per_n_summary = {}
    widths_by_n = []
    for n in n_values:
        mat = result.success_matrix(n)
        rmse = theory_rmse(result, n)
        row_widths = []
        threshold_rows = []
        for i, rho in enumerate(result.rho_grid):
            emp = empirical_threshold(result.delta_grid, mat[i])
            width = transition_width(result.delta_grid, mat[i])
            if width is not None:
                row_widths.append(width)
            threshold_rows.append({
                "rho": float(rho),
                "empirical_delta_50": emp,
                "theory_delta": phase_transition_delta(float(rho)),
                "transition_width": width,
            })
        mean_width = float(np.mean(row_widths)) if row_widths else None
        widths_by_n.append(mean_width if mean_width is not None else float("nan"))
        per_n_summary[str(n)] = {
            "rmse_vs_theory": rmse,
            "mean_transition_width": mean_width,
            "thresholds": threshold_rows,
        }
        print(f"n={n}: RMSE vs theory = {rmse:.4f}, "
              f"mean transition width = {mean_width}")

    summary = {
        "n_values": n_values,
        "delta_grid": delta_grid.tolist(),
        "rho_grid": rho_grid.tolist(),
        "trials_per_point": trials,
        "elapsed_seconds": elapsed,
        "per_n": per_n_summary,
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # --- figures ---
    plot_heatmaps(result, os.path.join(FIGURES_DIR, "phase_transition_heatmaps.png"))
    plot_threshold_curves(result, os.path.join(FIGURES_DIR, "threshold_curves.png"))
    plot_transition_width(n_values, widths_by_n,
                           os.path.join(FIGURES_DIR, "transition_width_vs_n.png"))
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
