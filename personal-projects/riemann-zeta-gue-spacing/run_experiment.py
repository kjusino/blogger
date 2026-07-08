#!/usr/bin/env python3
"""Entry point: runs the height sweep and the N-scaling sweep, writes
results/*.csv and figures/*.png.

Usage: python3 run_experiment.py
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import analyze_window, nested_subsample_metrics, loglog_fit
from src.plotting import (
    plot_spacing_histograms,
    plot_pair_correlation,
    plot_ks_vs_height,
    plot_scaling_vs_n,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

# Four non-overlapping windows spanning ~5 orders of magnitude in height T,
# each with the same window size N=300 so the height sweep isolates the
# effect of T alone.
HEIGHT_WINDOWS = [
    {"n_start": 1, "count": 300, "label": "n=1-300"},
    {"n_start": 10_000, "count": 300, "label": "n=10000-10300"},
    {"n_start": 50_000, "count": 300, "label": "n=50000-50300"},
    {"n_start": 300_000, "count": 300, "label": "n=300000-300300"},
]

# Nested prefixes of the n=10000 window, to check whether the residual
# gap to GUE shrinks at the finite-sample-noise rate as N grows.
N_SCALING_BASE_INDEX = 1  # the n=10000 window in HEIGHT_WINDOWS
N_SCALING_VALUES = [100, 200, 300]


def write_height_summary_csv(path, results):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "n_start", "count", "n_spacings", "t_min", "t_max",
            "mean_spacing", "std_spacing", "ks_gue", "p_gue", "ks_poisson",
            "p_poisson", "pc_err_montgomery", "pc_err_flat", "repulsion_fraction",
        ])
        for r in results:
            writer.writerow([
                r["label"], r["n_start"], r["count"], r["n_spacings"], r["t_min"], r["t_max"],
                r["mean_spacing"], r["std_spacing"], r["ks_gue"], r["p_gue"], r["ks_poisson"],
                r["p_poisson"], r["pc_err_montgomery"], r["pc_err_flat"], r["repulsion_fraction"],
            ])


def write_n_scaling_csv(path, results, fit):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_spacings", "ks_gue", "pc_err_montgomery", "repulsion_fraction"])
        for r in results:
            writer.writerow([r["n_spacings"], r["ks_gue"], r["pc_err_montgomery"], r["repulsion_fraction"]])
        writer.writerow([])
        writer.writerow(["loglog_slope", "stderr", "r_squared", "p_value"])
        writer.writerow([fit["slope"], fit["stderr"], fit["r_squared"], fit["p_value"]])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=== Height sweep: fixed N=300, four windows across ~5 orders of magnitude in T ===")
    height_results = []
    for spec in HEIGHT_WINDOWS:
        t0 = time.time()
        res = analyze_window(spec["n_start"], spec["count"], label=spec["label"])
        dt = time.time() - t0
        height_results.append(res)
        print(
            f"  {spec['label']:>18}  T≈[{res['t_min']:.0f}, {res['t_max']:.0f}]  "
            f"KS(GUE)={res['ks_gue']:.3f} (p={res['p_gue']:.3f})  "
            f"KS(Poisson)={res['ks_poisson']:.3f}  "
            f"repulsion(<0.2)={res['repulsion_fraction']:.3f}  [{dt:.1f}s]"
        )

    print("\n=== N-scaling sweep: nested prefixes of the n=10000 window ===")
    base_window = height_results[N_SCALING_BASE_INDEX]
    n_scaling_results = nested_subsample_metrics(base_window, N_SCALING_VALUES)
    for r in n_scaling_results:
        print(f"  N={r['n_spacings']:4d}  KS(GUE)={r['ks_gue']:.3f}  pc_err={r['pc_err_montgomery']:.3f}")

    ns = [r["n_spacings"] for r in n_scaling_results]
    ks = [r["ks_gue"] for r in n_scaling_results]
    fit = loglog_fit(ns, ks)
    print(
        f"\n  log-log fit of KS(GUE) vs N: slope={fit['slope']:.3f} +/- {fit['stderr']:.3f}, "
        f"R^2={fit['r_squared']:.3f} (finite-sample noise predicts slope=-0.5)"
    )

    write_height_summary_csv(os.path.join(RESULTS_DIR, "height_sweep_summary.csv"), height_results)
    write_n_scaling_csv(os.path.join(RESULTS_DIR, "n_scaling_summary.csv"), n_scaling_results, fit)

    plot_spacing_histograms(height_results, os.path.join(FIGURES_DIR, "spacing_histograms.png"))
    plot_pair_correlation(height_results, os.path.join(FIGURES_DIR, "pair_correlation.png"))
    plot_ks_vs_height(height_results, os.path.join(FIGURES_DIR, "ks_vs_height.png"))
    plot_scaling_vs_n(n_scaling_results, fit, os.path.join(FIGURES_DIR, "scaling_vs_n.png"))

    print("\nWrote results/*.csv and figures/*.png")


if __name__ == "__main__":
    main()
