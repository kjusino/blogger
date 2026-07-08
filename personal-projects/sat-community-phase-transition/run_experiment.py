#!/usr/bin/env python3
"""Reproduce the full experiment: sweep alpha x mu, solve instances, write
results/results.csv, and render all figures into figures/.

Usage: python run_experiment.py [--quick]

--quick runs a much smaller grid (for a fast smoke test); omit it to
reproduce the results reported in README.md.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_sweep, write_csv
from src.plots import generate_all_figures

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="tiny grid for a fast smoke test, not for real results")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        n_vars, alphas, mus, trials = 20, np.linspace(2.5, 6.5, 5).tolist(), [1.0, 0.0], 3
    else:
        n_vars, alphas, mus, trials = 80, np.linspace(2.5, 6.5, 21).tolist(), [1.0, 0.66, 0.33, 0.0], 25

    total = len(alphas) * len(mus)
    t_start = time.time()

    def progress(done, total, alpha, mu):
        elapsed = time.time() - t_start
        rate = elapsed / done
        eta = rate * (total - done)
        print(f"[{done}/{total}] finished alpha={alpha:.3f} mu={mu:.2f} "
              f"(elapsed={elapsed:.0f}s, eta={eta:.0f}s)", flush=True)

    print(f"Running sweep: n_vars={n_vars}, {len(alphas)} alphas x {len(mus)} mus x {trials} trials "
          f"= {len(alphas) * len(mus) * trials} solves")
    rows = run_sweep(alphas, mus, trials, n_vars=n_vars, seed=args.seed, progress=progress)

    results_dir = os.path.join(HERE, "results")
    figures_dir = os.path.join(HERE, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "results.csv")
    write_csv(rows, csv_path)
    print(f"Wrote {len(rows)} rows to {csv_path}")

    scatter_stats = generate_all_figures(rows, figures_dir)
    print(f"Wrote figures to {figures_dir}")

    with open(os.path.join(results_dir, "scatter_stats.json"), "w") as f:
        json.dump(scatter_stats, f, indent=2)

    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
