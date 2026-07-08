"""Figures, generated strictly from the CSVs in results/. Colors follow a
fixed categorical order (topology identity: RR=blue, ER=aqua, BA=orange) and
a fixed pair for the two competing theories (QMF=violet, HMF=red), so the
same entity keeps the same color across every figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOPOLOGY_COLOR = {"rr": "#2a78d6", "er": "#1baf7a", "ba": "#eb6834"}
TOPOLOGY_LABEL = {"rr": "Random regular", "er": "Erdos-Renyi", "ba": "Barabasi-Albert"}
QMF_COLOR = "#4a3aa7"
HMF_COLOR = "#e34948"
EMPIRICAL_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
MUTED = "#898781"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_degree_distributions(degree_samples: dict, path: Path) -> None:
    """degree_samples: {topology: np.ndarray of node degrees} at one N."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for topo, degrees in degree_samples.items():
        ax.hist(degrees, bins=np.arange(degrees.min(), degrees.max() + 2) - 0.5,
                alpha=0.55, color=TOPOLOGY_COLOR[topo], label=TOPOLOGY_LABEL[topo],
                density=True, zorder=2)
    _style_axes(ax)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Density")
    ax.set_title("Degree distributions at matched mean degree")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_susceptibility_grid(sweep_rows: list[dict], summaries: list[dict], ns: list[int],
                              topologies: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(topologies), len(ns), figsize=(4.2 * len(ns), 3.4 * len(topologies)),
                              squeeze=False)
    for i, topo in enumerate(topologies):
        for j, n in enumerate(ns):
            ax = axes[i][j]
            rows = [r for r in sweep_rows if r["topology"] == topo and int(r["n"]) == n]
            rows.sort(key=lambda r: r["tau"])
            taus = np.array([r["tau"] for r in rows])
            sus = np.array([r["susceptibility_mean"] for r in rows])
            sem = np.array([r["susceptibility_sem"] for r in rows])
            ax.plot(taus, sus, color=TOPOLOGY_COLOR[topo], linewidth=2, zorder=3)
            ax.fill_between(taus, sus - sem, sus + sem, color=TOPOLOGY_COLOR[topo], alpha=0.2, zorder=2)

            summary = next(s for s in summaries if s["topology"] == topo and int(s["n"]) == n)
            ax.axvline(summary["qmf_threshold"], color=QMF_COLOR, linestyle="--", linewidth=1.5,
                       label="QMF prediction", zorder=4)
            ax.axvline(summary["hmf_threshold"], color=HMF_COLOR, linestyle="--", linewidth=1.5,
                       label="HMF prediction", zorder=4)
            ax.axvline(summary["tau_c_empirical"], color=EMPIRICAL_COLOR, linestyle="-", linewidth=1.5,
                       label="Empirical peak", zorder=5)

            _style_axes(ax)
            ax.set_title(f"{TOPOLOGY_LABEL[topo]}, N={n}", fontsize=10)
            if i == len(topologies) - 1:
                ax.set_xlabel("tau = beta/delta")
            if j == 0:
                ax.set_ylabel("Susceptibility chi")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle("SIS susceptibility vs. effective spreading rate: QMF/HMF predictions vs. QS-measured peak")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_comparison(summaries: list[dict], path: Path) -> None:
    labels = [f"{TOPOLOGY_LABEL[s['topology']]}, N={s['n']}" for s in summaries]
    x = np.arange(len(summaries))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, [s["tau_c_empirical"] for s in summaries],
                yerr=[s["tau_c_sem"] for s in summaries],
                fmt="o", color=EMPIRICAL_COLOR, label="Empirical (QS peak)", capsize=3, zorder=5)
    ax.scatter(x, [s["qmf_threshold"] for s in summaries], marker="^", color=QMF_COLOR,
               label="QMF: 1/lambda_max", s=60, zorder=4)
    ax.scatter(x, [s["hmf_threshold"] for s in summaries], marker="s", color=HMF_COLOR,
               label="HMF: <k>/<k^2>", s=60, zorder=4)

    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Epidemic threshold tau_c")
    ax.set_yscale("log")
    ax.set_title("Empirical threshold vs. QMF and HMF predictions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_heterogeneity_vs_gap(summaries: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s in summaries:
        ax.scatter(s["heterogeneity_ratio"], s["gap_hmf_minus_qmf"],
                   color=TOPOLOGY_COLOR[s["topology"]], s=70, zorder=3,
                   label=TOPOLOGY_LABEL[s["topology"]])

    het = np.array([s["heterogeneity_ratio"] for s in summaries])
    gap = np.array([s["gap_hmf_minus_qmf"] for s in summaries])
    valid = ~np.isnan(gap)
    if valid.sum() >= 2:
        coeffs = np.polyfit(het[valid], gap[valid], 1)
        xs = np.linspace(het.min(), het.max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color=MUTED, linestyle="--", linewidth=1.5, zorder=2,
                label="linear trend")

    ax.axhline(0, color=GRID_COLOR, linewidth=1, zorder=1)
    _style_axes(ax)
    ax.set_xlabel("Heterogeneity ratio  <k^2> / <k>^2")
    ax.set_ylabel("eps_HMF - eps_QMF  (positive = QMF more accurate)")
    ax.set_title("Does degree heterogeneity predict which theory wins?")

    handles, plot_labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, plot_labels):
        seen[l] = h
    ax.legend(seen.values(), seen.keys(), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
