#!/usr/bin/env python3
"""Run the full price-of-anarchy topology-independence study and write
results (CSV + JSON summary) to results/ and figures to figures/.

Usage:
    python run_experiment.py            # full battery (a couple minutes)
    python run_experiment.py --quick    # small battery, for a fast smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from src.experiment import run_topology_battery, run_worst_case_convergence
from src.network import braess_network
from src.plots import (
    plot_braess_example,
    plot_convergence_two_link,
    plot_poa_distributions,
    plot_poa_vs_degree,
)
from src.solvers import price_of_anarchy
from src.theory import poa_bound

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Small battery / few trials, for a fast smoke test.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    degrees = [1, 2, 3, 4]
    if args.quick:
        n_trials, max_edges = 15, 8
        b_grid = np.linspace(0.01, 3.0, 20)
    else:
        n_trials, max_edges = 300, 20
        b_grid = np.linspace(0.001, 5.0, 200)

    print(f"Topology battery: degrees={degrees}, {n_trials} random series-parallel "
          f"networks/degree (up to {max_edges} edges each) = "
          f"{len(degrees) * n_trials} networks, {len(degrees) * n_trials * 2} NLP solves")

    t0 = time.time()
    battery = run_topology_battery(degrees, n_trials=n_trials, max_edges=max_edges, seed=2024)
    battery_elapsed = time.time() - t0
    print(f"Topology battery finished in {battery_elapsed:.1f}s")

    t0 = time.time()
    convergence = run_worst_case_convergence(degrees, b_grid=b_grid)
    convergence_elapsed = time.time() - t0
    print(f"Worst-case two-link sweep finished in {convergence_elapsed:.1f}s "
          f"({len(degrees) * len(b_grid)} NLP solves)")

    # --- CSV of every topology-battery trial ---
    csv_path = os.path.join(RESULTS_DIR, "topology_battery.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "n_edges", "poa", "theoretical_bound"])
        for p, result in battery.items():
            for trial in result.trials:
                writer.writerow([trial.degree, trial.n_edges, trial.poa, result.bound])
    print(f"Wrote {csv_path}")

    # --- CSV of the worst-case convergence sweep ---
    conv_csv_path = os.path.join(RESULTS_DIR, "worst_case_convergence.csv")
    with open(conv_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "b", "poa", "theoretical_bound"])
        for p, points in convergence.items():
            for pt in points:
                writer.writerow([p, pt.b, pt.poa, poa_bound(p)])
    print(f"Wrote {conv_csv_path}")

    # --- Braess's paradox sanity example ---
    r_no = price_of_anarchy(braess_network(with_shortcut=False))
    r_yes = price_of_anarchy(braess_network(with_shortcut=True))

    # --- JSON summary ---
    summary = {
        "degrees": degrees,
        "n_trials_per_degree": n_trials,
        "max_edges": max_edges,
        "battery_elapsed_seconds": battery_elapsed,
        "convergence_elapsed_seconds": convergence_elapsed,
        "per_degree": {
            str(p): {
                "theoretical_bound": result.bound,
                "max_empirical_poa": result.max_poa,
                "mean_empirical_poa": float(np.mean(result.poas)),
                "n_violations": result.n_violations,
                "worst_case_two_link_peak_poa": max(pt.poa for pt in convergence[p]),
            }
            for p, result in battery.items()
        },
        "braess_paradox": {
            "cost_without_shortcut": r_no["equilibrium_cost"],
            "cost_with_shortcut": r_yes["equilibrium_cost"],
        },
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    for p, stats in summary["per_degree"].items():
        print(f"degree={p}: bound={stats['theoretical_bound']:.4f}, "
              f"max empirical PoA={stats['max_empirical_poa']:.4f}, "
              f"violations={stats['n_violations']}, "
              f"two-link peak={stats['worst_case_two_link_peak_poa']:.4f}")

    # --- figures ---
    convergence_peaks = {p: pts for p, pts in convergence.items()}
    plot_poa_vs_degree(battery, convergence_peaks, os.path.join(FIGURES_DIR, "poa_vs_degree.png"))
    plot_poa_distributions(battery, os.path.join(FIGURES_DIR, "poa_distributions.png"))
    plot_convergence_two_link({p: convergence[p] for p in degrees},
                               os.path.join(FIGURES_DIR, "convergence_two_link.png"))
    plot_braess_example(r_no["equilibrium_cost"], r_yes["equilibrium_cost"],
                         os.path.join(FIGURES_DIR, "braess_paradox.png"))
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
