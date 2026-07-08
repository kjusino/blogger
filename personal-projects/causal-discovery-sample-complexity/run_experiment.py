#!/usr/bin/env python3
"""Entry point: runs both scaling sweeps (log p and d^2), writes results to
results/*.csv and figures to figures/*.png.

Usage: python3 run_experiment.py
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_p_sweep, run_d_sweep
from src.plotting import plot_recovery_curves, plot_scaling_fit, plot_rescaled_collapse

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

SEED = 20260708
D_FIXED = 2
P_VALUES = [10, 20, 35, 55, 80]
P_FIXED = 14
D_VALUES = [1, 2, 3]
MIN_MARGIN = 0.10
COARSE_TRIALS = 8
FINE_TRIALS = 20
FINE_POINTS = 6


def write_curve_csv(path, configs, varying_key):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "d", "alpha", "n", "trials", "recovery_prob", "mean_shd", "mean_precision", "mean_recall"])
        for cfg in configs:
            for row in cfg["rows"]:
                writer.writerow([
                    row["p"], row["d"], row["alpha"], row["n"], row["trials"],
                    row["recovery_prob"], row["mean_shd"], row["mean_precision"], row["mean_recall"],
                ])


def write_summary_csv(path, configs, varying_key):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "d", "alpha", "n50_interp", "n50_logistic", "logistic_steepness"])
        for cfg in configs:
            writer.writerow([
                cfg["p"], cfg["d"], cfg["alpha"], cfg["n50_interp"],
                cfg["n50_logistic"], cfg["logistic_steepness"],
            ])


def write_fit_csv(path, fit, x_name):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_variable", "slope", "stderr", "intercept", "r_squared", "p_value"])
        writer.writerow([x_name, fit["slope"], fit["stderr"], fit["intercept"], fit["r_squared"], fit["p_value"]])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"=== Sweep A: n50 vs p (d fixed at {D_FIXED}), p in {P_VALUES} ===")
    p_configs, p_fit = run_p_sweep(
        P_VALUES, D_FIXED, rng, min_margin=MIN_MARGIN,
        coarse_trials=COARSE_TRIALS, fine_trials=FINE_TRIALS, fine_points=FINE_POINTS,
    )
    for cfg in p_configs:
        print(f"  p={cfg['p']:3d} d={cfg['d']} alpha={cfg['alpha']:.2e} n50={cfg['n50_interp']:.0f}")
    print(f"  log-log fit: slope={p_fit['slope']:.3f} +/- {p_fit['stderr']:.3f}, R^2={p_fit['r_squared']:.3f} "
          f"(theory predicts slope=1, i.e. n50 ~ log(p))")

    print(f"\n=== Sweep B: n50 vs d (p fixed at {P_FIXED}), d in {D_VALUES} ===")
    d_configs, d_fit = run_d_sweep(
        D_VALUES, P_FIXED, rng, min_margin=MIN_MARGIN,
        coarse_trials=COARSE_TRIALS, fine_trials=FINE_TRIALS, fine_points=FINE_POINTS,
    )
    for cfg in d_configs:
        print(f"  d={cfg['d']} p={cfg['p']:3d} alpha={cfg['alpha']:.2e} n50={cfg['n50_interp']:.0f}")
    print(f"  log-log fit: slope={d_fit['slope']:.3f} +/- {d_fit['stderr']:.3f}, R^2={d_fit['r_squared']:.3f} "
          f"(theory predicts slope=2, i.e. n50 ~ d^2)")

    write_curve_csv(os.path.join(RESULTS_DIR, "p_sweep_curves.csv"), p_configs, "p")
    write_curve_csv(os.path.join(RESULTS_DIR, "d_sweep_curves.csv"), d_configs, "d")
    write_summary_csv(os.path.join(RESULTS_DIR, "p_sweep_summary.csv"), p_configs, "p")
    write_summary_csv(os.path.join(RESULTS_DIR, "d_sweep_summary.csv"), d_configs, "d")
    write_fit_csv(os.path.join(RESULTS_DIR, "p_sweep_loglog_fit.csv"), p_fit, "p")
    write_fit_csv(os.path.join(RESULTS_DIR, "d_sweep_loglog_fit.csv"), d_fit, "d")

    plot_recovery_curves(
        p_configs, "p", f"d = {D_FIXED} (fixed)", os.path.join(FIGURES_DIR, "p_sweep_recovery_curves.png"),
        "Skeleton recovery probability vs sample size, varying p",
    )
    plot_recovery_curves(
        d_configs, "d", f"p = {P_FIXED} (fixed)", os.path.join(FIGURES_DIR, "d_sweep_recovery_curves.png"),
        "Skeleton recovery probability vs sample size, varying d",
    )
    plot_scaling_fit(
        p_configs, "p", os.path.join(FIGURES_DIR, "p_sweep_scaling_fit.png"),
        "Sample complexity n50 vs number of variables p", "p (log scale)", p_fit,
    )
    plot_scaling_fit(
        d_configs, "d", os.path.join(FIGURES_DIR, "d_sweep_scaling_fit.png"),
        "Sample complexity n50 vs max degree d", "d (log scale)", d_fit,
    )
    plot_rescaled_collapse(
        p_configs, "p", np.log, "log(p)", os.path.join(FIGURES_DIR, "p_sweep_collapse.png"),
        "Rescaled collapse: n / log(p)", f"d = {D_FIXED} (fixed)",
    )
    plot_rescaled_collapse(
        d_configs, "d", lambda d: d ** 2, "d^2", os.path.join(FIGURES_DIR, "d_sweep_collapse.png"),
        "Rescaled collapse: n / d^2", f"p = {P_FIXED} (fixed)",
    )

    print("\nWrote results/*.csv and figures/*.png")


if __name__ == "__main__":
    main()
