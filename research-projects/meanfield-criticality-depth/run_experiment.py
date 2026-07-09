#!/usr/bin/env python3
"""Top-level entry point: runs the full (sigma_w2, sigma_b2) grid sweep,
writes results/grid_results.csv and results/summary.json, and regenerates
all figures in figures/. Deterministic given the fixed seed below.

    python run_experiment.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from experiment import run_grid_sweep, write_csv, write_json, summarize
import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

SEED = 0
SIGMA_W2_GRID = np.round(np.linspace(0.5, 3.5, 7), 4)
SIGMA_B2_GRID = np.round(np.linspace(0.0, 0.5, 7), 4)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t0 = time.time()
    rows = run_grid_sweep(SIGMA_W2_GRID, SIGMA_B2_GRID, seed=SEED)
    elapsed = time.time() - t0
    print(f"\nGrid sweep finished in {elapsed:.1f}s over {len(rows)} points")

    write_csv(rows, os.path.join(RESULTS_DIR, "grid_results.csv"))

    summary = summarize(rows)
    summary["grid"] = {
        "sigma_w2": SIGMA_W2_GRID.tolist(),
        "sigma_b2": SIGMA_B2_GRID.tolist(),
        "seed": SEED,
        "elapsed_seconds": elapsed,
    }
    write_json(summary, os.path.join(RESULTS_DIR, "summary.json"))
    for k, v in summary.items():
        print(k, ":", v)

    plotting.plot_correlation_maps(os.path.join(FIGURES_DIR, "correlation_maps.png"))
    plotting.plot_signal_propagation(os.path.join(FIGURES_DIR, "signal_propagation_examples.png"), seed=SEED)
    plotting.plot_phase_diagram(rows, SIGMA_W2_GRID, SIGMA_B2_GRID, os.path.join(FIGURES_DIR, "phase_diagram.png"))
    plotting.plot_depth_vs_xi(rows, os.path.join(FIGURES_DIR, "depth_vs_correlation_length.png"))
    print("\nFigures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
