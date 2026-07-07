#!/usr/bin/env python3
"""Run the full (or --quick smoke-test) experiment and save results/figures.

Full mode:
    python3 run_experiment.py

Quick smoke test (small n, few trials, seconds not minutes):
    python3 run_experiment.py --quick
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import (
    connected_fraction_curve,
    ratio_summary,
    run_betti1_curve,
    run_threshold_experiment,
)
from src.scaling import bootstrap_ci, transition_width
from src.theory import penrose_threshold_radius

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def bootstrap_transition_width(thresholds, theory_r_c, rng, n_boot=2000, alpha=0.05):
    """Bootstrap CI for the 10%-90% transition width, resampling raw per-trial
    thresholds (each an exact longest-MST-edge draw) with replacement."""
    n_trials = len(thresholds)
    r_grid = np.linspace(0, theory_r_c * 3, 400)
    idx = rng.integers(0, n_trials, size=(n_boot, n_trials))
    widths = np.empty(n_boot)
    for b in range(n_boot):
        frac = connected_fraction_curve(thresholds[idx[b]], r_grid)
        widths[b], _, _ = transition_width(r_grid, frac)
    point, _, _ = transition_width(r_grid, connected_fraction_curve(thresholds, r_grid))
    lo, hi = np.quantile(widths, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke-test sizes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.quick:
        n_values = [20, 40]
        trials_per_n = 10
        betti1_n_values = [20]
        betti1_r_points = 6
        betti1_trials = 3
        width_boot = 200
    else:
        n_values = [50, 100, 200, 400, 800]
        trials_per_n = 300
        betti1_n_values = [80, 320]
        betti1_r_points = 25
        betti1_trials = 20
        width_boot = 2000

    t0 = time.time()
    print(f"Running threshold experiment for n={n_values}, {trials_per_n} trials each...")
    threshold_results = run_threshold_experiment(n_values, trials_per_n, rng)

    summary_rows = []
    for n in n_values:
        result = threshold_results[n]
        summary = ratio_summary(result)
        _, lo, hi = bootstrap_ci(result.thresholds, rng, statistic=np.mean)
        ratio_lo = lo / result.theory_r_c
        ratio_hi = hi / result.theory_r_c
        r_grid = np.linspace(0, result.theory_r_c * 3, 400)
        frac = connected_fraction_curve(result.thresholds, r_grid)
        width, r_low, r_high = transition_width(r_grid, frac)
        width_point, width_lo, width_hi = bootstrap_transition_width(
            result.thresholds, result.theory_r_c, rng, n_boot=width_boot
        )
        summary_rows.append(
            {
                "n": n,
                "trials": trials_per_n,
                "mean_threshold": summary["mean_threshold"],
                "mean_threshold_ci_lo": lo,
                "mean_threshold_ci_hi": hi,
                "theory_r_c": summary["theory_r_c"],
                "ratio": summary["ratio"],
                "ratio_ci_lo": ratio_lo,
                "ratio_ci_hi": ratio_hi,
                "relative_error_pct": 100 * abs(summary["ratio"] - 1.0),
                "transition_width": width,
                "transition_width_normalized": width / result.theory_r_c,
                "transition_width_normalized_ci_lo": width_lo / result.theory_r_c,
                "transition_width_normalized_ci_hi": width_hi / result.theory_r_c,
            }
        )
        print(
            f"  n={n:4d}  ratio={summary['ratio']:.4f} "
            f"(95% CI [{ratio_lo:.4f}, {ratio_hi:.4f}])  "
            f"transition_width/r_c={width / result.theory_r_c:.4f} "
            f"(95% CI [{width_lo / result.theory_r_c:.4f}, {width_hi / result.theory_r_c:.4f}])"
        )

    with open(RESULTS_DIR / "threshold_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    with open(RESULTS_DIR / "raw_thresholds.json", "w") as f:
        json.dump({str(n): threshold_results[n].thresholds.tolist() for n in n_values}, f, indent=2)

    print(f"\nRunning Betti_1 sweep for n={betti1_n_values} ({betti1_r_points} r-points, "
          f"{betti1_trials} trials each)...")
    betti1_results = {}
    for n in betti1_n_values:
        r_c = penrose_threshold_radius(n)
        r_grid = np.linspace(0.5 * r_c, 3.0 * r_c, betti1_r_points)
        result = run_betti1_curve(n, r_grid, betti1_trials, rng)
        betti1_results[n] = result
        peak_idx = int(np.argmax(result.mean_betti1))
        print(
            f"  n={n:4d}  peak mean Betti_1={result.mean_betti1[peak_idx]:.3f} "
            f"at r/r_c={r_grid[peak_idx] / r_c:.3f}"
        )

    with open(RESULTS_DIR / "betti1_curves.json", "w") as f:
        json.dump(
            {
                str(n): {
                    "r_grid": res.r_grid.tolist(),
                    "r_over_rc": (res.r_grid / penrose_threshold_radius(n)).tolist(),
                    "mean_betti0": res.mean_betti0.tolist(),
                    "mean_betti1": res.mean_betti1.tolist(),
                    "frac_connected": res.frac_connected.tolist(),
                }
                for n, res in betti1_results.items()
            },
            f,
            indent=2,
        )

    make_figures(summary_rows, threshold_results, betti1_results, n_values)
    print(f"\nDone in {time.time() - t0:.1f}s. Results in {RESULTS_DIR}, figures in {FIGURES_DIR}.")


def make_figures(summary_rows, threshold_results, betti1_results, n_values):
    ns = [row["n"] for row in summary_rows]
    ratios = [row["ratio"] for row in summary_rows]
    ratio_los = [row["ratio_ci_lo"] for row in summary_rows]
    ratio_his = [row["ratio_ci_hi"] for row in summary_rows]

    # Figure 1: empirical/theoretical threshold ratio -> 1
    fig, ax = plt.subplots(figsize=(6, 4.5))
    yerr = [
        [r - lo for r, lo in zip(ratios, ratio_los)],
        [hi - r for r, hi in zip(ratios, ratio_his)],
    ]
    ax.errorbar(ns, ratios, yerr=yerr, fmt="o-", capsize=4, label="empirical / theory")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Penrose theory (ratio = 1)")
    ax.set_xscale("log")
    ax.set_xlabel("n (number of points)")
    ax.set_ylabel(r"$r_{\mathrm{empirical}} / r_c(n)$")
    ax.set_title("Connectivity threshold: empirical vs. Penrose's theory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_ratio_vs_theory.png", dpi=150)
    plt.close(fig)

    # Figure 2: connected-fraction curves rescaled by r_c(n) -- sharpening transition
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in n_values:
        result = threshold_results[n]
        r_grid = np.linspace(0, result.theory_r_c * 3, 400)
        frac = connected_fraction_curve(result.thresholds, r_grid)
        ax.plot(r_grid / result.theory_r_c, frac, label=f"n={n}")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label=r"theory $r_c(n)$")
    ax.set_xlabel(r"$r / r_c(n)$")
    ax.set_ylabel("P(connected)")
    ax.set_title("Connectivity transition sharpens as n grows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "connectivity_transition.png", dpi=150)
    plt.close(fig)

    # Figure 3: transition width (normalized) vs n, with bootstrap CI band
    widths = [row["transition_width_normalized"] for row in summary_rows]
    width_los = [row["transition_width_normalized_ci_lo"] for row in summary_rows]
    width_his = [row["transition_width_normalized_ci_hi"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    w_yerr = [
        [w - lo for w, lo in zip(widths, width_los)],
        [hi - w for w, hi in zip(widths, width_his)],
    ]
    ax.errorbar(ns, widths, yerr=w_yerr, fmt="o-", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel("n (number of points)")
    ax.set_ylabel(r"transition width / $r_c(n)$ (10%-90% crossing)")
    ax.set_title("Transition sharpness vs. system size (bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "transition_width_vs_n.png", dpi=150)
    plt.close(fig)

    # Figure 4: Betti_1 curves -- the homological connectivity window
    fig, axes = plt.subplots(1, len(betti1_results), figsize=(6 * len(betti1_results), 4.5), squeeze=False)
    for ax, (n, res) in zip(axes[0], betti1_results.items()):
        r_c = penrose_threshold_radius(n)
        x = res.r_grid / r_c
        ax2 = ax.twinx()
        ax.plot(x, res.mean_betti1, "o-", color="tab:red", label=r"mean $\beta_1$ (cycles)")
        ax2.plot(x, res.frac_connected, "s--", color="tab:blue", alpha=0.6, label="P(connected)")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel(r"$r / r_c(n)$")
        ax.set_ylabel(r"mean $\beta_1$", color="tab:red")
        ax2.set_ylabel("P(connected)", color="tab:blue")
        ax.set_title(f"n={n}")
    fig.suptitle("Homological connectivity window: cycles appear, then vanish, above the graph threshold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "betti1_window.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
