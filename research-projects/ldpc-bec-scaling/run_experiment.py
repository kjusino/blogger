#!/usr/bin/env python3
"""Run the full LDPC-over-BEC finite-length scaling experiment.

Usage:
    python3 run_experiment.py            # full grid (~30-60s)
    python3 run_experiment.py --quick    # smoke-test grid (~2s)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.decoder import peel_decode, sample_erasures
from src.density_evolution import de_trajectory, find_threshold
from src.experiment import fit_power_law, measure_bler_curve
from src.tanner import build_regular_tanner_graph

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"

DV, DC = 3, 6
SEED = 2024


def run(quick: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = random.Random(SEED)

    eps_star = find_threshold(DV, DC)
    print(f"({DV},{DC})-regular ensemble BEC threshold (density evolution): {eps_star:.6f}")

    if quick:
        blocklengths = [64, 128, 256]
        n_graph_instances, coarse_trials, fine_trials = 2, 10, 20
    else:
        blocklengths = [128, 256, 512, 1024, 2048, 4096]
        n_graph_instances, coarse_trials, fine_trials = 3, 40, 150

    curves = []
    for n in blocklengths:
        curve = measure_bler_curve(
            n,
            DV,
            DC,
            eps_star,
            rng,
            n_graph_instances=n_graph_instances,
            coarse_trials=coarse_trials,
            fine_trials=fine_trials,
        )
        curves.append(curve)
        print(
            f"  n={n:5d}  eps50={curve.eps50:.5f}  "
            f"gap={eps_star - curve.eps50:.5f}  width90/10={curve.width_90_10:.5f}"
        )

    with open(RESULTS_DIR / "bler_points.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "epsilon", "bler", "trials"])
        for curve in curves:
            for eps, bler, trials in zip(curve.epsilons, curve.blers, curve.trials):
                writer.writerow([curve.n, eps, bler, trials])

    with open(RESULTS_DIR / "curve_fits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "eps50", "steepness_k", "width_90_10", "gap_to_threshold"])
        for curve in curves:
            writer.writerow(
                [curve.n, curve.eps50, curve.steepness, curve.width_90_10, eps_star - curve.eps50]
            )

    ns = [c.n for c in curves]
    gaps = [eps_star - c.eps50 for c in curves]
    widths = [c.width_90_10 for c in curves]

    gap_c, gap_exponent = fit_power_law(ns, gaps)
    width_c, width_exponent = fit_power_law(ns, widths)

    summary = {
        "ensemble": {"dv": DV, "dc": DC, "rate": 1 - DV / DC},
        "de_threshold": eps_star,
        "blocklengths": blocklengths,
        "shift_scaling": {
            "fitted_prefactor_C": gap_c,
            "fitted_exponent": gap_exponent,
            "theory_exponent": -2 / 3,
        },
        "width_scaling": {
            "fitted_prefactor_C": width_c,
            "fitted_exponent": width_exponent,
            "theory_exponent": -1 / 2,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nGap (eps* - eps50) ~ n^p:   fitted p = {gap_exponent:.4f}  (theory: -0.6667)")
    print(f"Width (eps90-eps10) ~ n^q:  fitted q = {width_exponent:.4f}  (theory: -0.5000)")

    _make_figures(curves, eps_star, ns, gaps, widths, gap_c, gap_exponent, width_c, width_exponent)


def _make_figures(curves, eps_star, ns, gaps, widths, gap_c, gap_exponent, width_c, width_exponent):
    # Figure 1: DE trajectories below/at/above threshold
    fig, ax = plt.subplots(figsize=(7, 5))
    for eps, label in [
        (eps_star * 0.85, "below threshold"),
        (eps_star, "at threshold"),
        (eps_star * 1.15, "above threshold"),
    ]:
        traj = de_trajectory(eps, DV, DC, n_iters=60)
        ax.plot(traj, label=f"$\\epsilon$={eps:.3f} ({label})")
    ax.set_xlabel("DE iteration $\\ell$")
    ax.set_ylabel("edge erasure probability $x_\\ell$")
    ax.set_title(f"Density evolution, ({DV},{DC})-regular BEC ensemble")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "density_evolution_trajectories.png", dpi=150)
    plt.close(fig)

    # Figure 2: BLER waterfall curves for each n
    fig, ax = plt.subplots(figsize=(7, 5))
    for curve in curves:
        order = np.argsort(curve.epsilons)
        eps_sorted = np.array(curve.epsilons)[order]
        bler_sorted = np.array(curve.blers)[order]
        ax.plot(eps_sorted, bler_sorted, "o", markersize=3, label=f"n={curve.n}")
    ax.axvline(eps_star, color="black", linestyle="--", label=f"DE threshold={eps_star:.4f}")
    ax.set_xlabel("erasure probability $\\epsilon$")
    ax.set_ylabel("block error rate")
    ax.set_title("Finite-length waterfall curves vs. asymptotic threshold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "waterfall_curves.png", dpi=150)
    plt.close(fig)

    # Figure 3: gap scaling (log-log)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, gaps, "o", label="measured gap $\\epsilon^*-\\epsilon_{50}(n)$")
    n_fit = np.linspace(min(ns), max(ns), 200)
    ax.loglog(n_fit, gap_c * n_fit**gap_exponent, "-", label=f"fit: exponent={gap_exponent:.3f}")
    ax.loglog(n_fit, gaps[0] * (n_fit / ns[0]) ** (-2 / 3), "--", label="theory: exponent=-2/3")
    ax.set_xlabel("blocklength $n$")
    ax.set_ylabel("$\\epsilon^* - \\epsilon_{50}(n)$")
    ax.set_title("Threshold-distance shift scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shift_scaling.png", dpi=150)
    plt.close(fig)

    # Figure 4: width scaling (log-log)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, widths, "o", label="measured width $\\epsilon_{90}-\\epsilon_{10}$")
    ax.loglog(
        n_fit, width_c * n_fit**width_exponent, "-", label=f"fit: exponent={width_exponent:.3f}"
    )
    ax.loglog(n_fit, widths[0] * (n_fit / ns[0]) ** (-1 / 2), "--", label="theory: exponent=-1/2")
    ax.set_xlabel("blocklength $n$")
    ax.set_ylabel("waterfall width $\\epsilon_{90}-\\epsilon_{10}$")
    ax.set_title("Waterfall transition-width scaling")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "width_scaling.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke-test grid")
    args = parser.parse_args()
    run(quick=args.quick)
