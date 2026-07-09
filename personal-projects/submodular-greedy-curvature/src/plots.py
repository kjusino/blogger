"""Figure generation for the curvature-vs-greedy-ratio experiment."""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.theory import WORST_CASE_BOUND, curvature_bound

COLORS = {
    "curvature_bound": "#1b6ca8",
    "worst_case_bound": "#c0392b",
    "points": "#2e2e2e",
    "highlight": "#27ae60",
}


def _curve_xy(n=400):
    xs = np.linspace(0, 1, n)
    ys = [curvature_bound(x) for x in xs]
    return xs, ys


def plot_ratio_vs_curvature(records, out_path):
    curvatures = np.array([r["curvature"] for r in records])
    ratios = np.array([r["ratio"] for r in records])
    k_over_n = np.array([r["k_over_n"] for r in records])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(curvatures, ratios, c=k_over_n, cmap="viridis",
                     s=22, alpha=0.75, edgecolors="none", label="observed trial")
    xs, ys = _curve_xy()
    ax.plot(xs, ys, color=COLORS["curvature_bound"], lw=2.2,
             label=r"curvature bound $(1-e^{-c})/c$")
    ax.axhline(WORST_CASE_BOUND, color=COLORS["worst_case_bound"], lw=2,
               linestyle="--", label=r"worst-case bound $1-1/e$")
    ax.set_xlabel("total curvature c")
    ax.set_ylabel("realized ratio  greedy(S) / OPT")
    ax.set_title("Greedy's realized ratio vs. instance curvature")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(0.55, ratios.min() - 0.03), 1.03)
    ax.legend(loc="lower left", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("k / n")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_validity_histogram(records, out_path):
    curvatures = np.array([r["curvature"] for r in records])
    ratios = np.array([r["ratio"] for r in records])
    bounds = np.array([curvature_bound(c) for c in curvatures])
    slack = ratios - bounds

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(slack, bins=40, color=COLORS["curvature_bound"], alpha=0.85)
    ax.axvline(0.0, color=COLORS["worst_case_bound"], lw=2, linestyle="--",
               label="theorem boundary (slack = 0)")
    ax.set_xlabel("ratio − curvature_bound(c)  (slack)")
    ax.set_ylabel("count")
    ax.set_title("M1: the curvature bound is never violated\n(all slack values are ≥ 0)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_vs_curvature(records, out_path):
    curvatures = np.array([r["curvature"] for r in records])
    gap = 1.0 - np.array([r["ratio"] for r in records])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(curvatures, gap, color=COLORS["points"], s=20, alpha=0.6)
    if len(curvatures) > 1 and np.std(curvatures) > 0:
        coeffs = np.polyfit(curvatures, gap, 1)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(coeffs, xs), color=COLORS["worst_case_bound"], lw=2,
                label=f"linear fit (slope={coeffs[0]:.3f})")
        ax.legend(fontsize=9)
    ax.set_xlabel("total curvature c")
    ax.set_ylabel("gap from optimal  (1 − ratio)")
    ax.set_title("M2: does curvature predict how far greedy falls short?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mae_comparison(summary, out_path):
    m3 = summary["m3_superior_predictor"]
    labels = ["curvature bound\n(1-e^-c)/c", "constant bound\n1-1/e"]
    values = [m3["mae_curvature_bound"], m3["mae_constant_bound"]]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bars = ax.bar(labels, values, color=[COLORS["curvature_bound"], COLORS["worst_case_bound"]])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=10)
    ax.set_ylabel("mean |min_ratio(instance) − bound|")
    ax.set_title("M3: which bound better characterizes\neach instance's worst-observed ratio?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_extremes_boxplot(records, out_path):
    curvatures = np.array([r["curvature"] for r in records])
    ratios = np.array([r["ratio"] for r in records])

    low = ratios[curvatures < 0.05]
    mid = ratios[(curvatures >= 0.05) & (curvatures <= 0.9)]
    high = ratios[curvatures > 0.9]

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [d for d in (low, mid, high) if len(d) > 0]
    tick_labels = [lbl for lbl, d in zip(
        ["near-modular\n(c < 0.05)", "mid curvature\n(0.05–0.9)", "near-degenerate\n(c > 0.9)"],
        (low, mid, high)) if len(d) > 0]
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["curvature_bound"])
        patch.set_alpha(0.6)
    ax.axhline(WORST_CASE_BOUND, color=COLORS["worst_case_bound"], lw=1.5,
               linestyle="--", label=r"worst-case bound $1-1/e$")
    ax.axhline(1.0, color=COLORS["highlight"], lw=1.5, linestyle=":",
               label="optimal (ratio = 1)")
    ax.set_ylabel("realized ratio")
    ax.set_title("M4: realized ratio at curvature extremes")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ratio_vs_redundancy_mult(records, out_path):
    """Mean +/- range of the realized ratio as a function of the design
    knob (redundancy_mult), on a log x-axis. Reveals whether difficulty
    increases monotonically with the knob or peaks at an intermediate
    value even as measured curvature keeps climbing toward 1."""
    by_mult = {}
    for r in records:
        by_mult.setdefault(r["redundancy_mult"], []).append(r["ratio"])
    mults = sorted(by_mult.keys())
    means = [np.mean(by_mult[m]) for m in mults]
    mins = [np.min(by_mult[m]) for m in mults]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [m if m > 0 else 1e-3 for m in mults]  # log-scale placeholder for 0
    ax.plot(x, means, "o-", color=COLORS["curvature_bound"], label="mean ratio")
    ax.plot(x, mins, "x--", color=COLORS["worst_case_bound"], label="min ratio")
    ax.set_xscale("log")
    ax.set_xlabel("redundancy_mult (design knob, log scale; leftmost point is mult=0)")
    ax.set_ylabel("realized ratio")
    ax.set_title("Realized ratio vs. redundancy intensity\n(is the hardest case at the extreme, or in between?)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_figures(records, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_ratio_vs_curvature(records, os.path.join(out_dir, "ratio_vs_curvature.png"))
    plot_validity_histogram(records, os.path.join(out_dir, "validity_histogram.png"))
    plot_gap_vs_curvature(records, os.path.join(out_dir, "gap_vs_curvature.png"))
    plot_mae_comparison(summary, os.path.join(out_dir, "mae_comparison.png"))
    plot_extremes_boxplot(records, os.path.join(out_dir, "extremes_boxplot.png"))
    plot_ratio_vs_redundancy_mult(records, os.path.join(out_dir, "ratio_vs_redundancy_mult.png"))
