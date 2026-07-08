"""Figure generation. Headless (Agg) backend so this runs without a display."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.theory import poa_bound

COLORS = ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#c53030", "#1a202c"]


def plot_poa_vs_degree(battery_results: dict, convergence_peaks: dict, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    p_smooth = np.linspace(1, max(battery_results) + 0.05, 400)
    ax.plot(p_smooth, [poa_bound(p) for p in p_smooth], "-", color="#1a202c",
             linewidth=2, label=r"theory: $\beta(p) = 1/(1-p(p+1)^{-(p+1)/p})$", zorder=2)

    degrees = sorted(battery_results)
    max_poas = [battery_results[p].max_poa for p in degrees]
    ax.scatter(degrees, max_poas, marker="o", s=70, color="#2b6cb0", zorder=4,
               label="max empirical PoA, random topologies")

    peak_degrees = sorted(convergence_peaks)
    peak_vals = [max(pt.poa for pt in convergence_peaks[p]) for p in peak_degrees]
    ax.scatter(peak_degrees, peak_vals, marker="^", s=90, color="#c05621", zorder=5,
               label="worst-case two-link network (generic solver)")

    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel("price of anarchy")
    ax.set_title("PoA bound is topology-independent: random topologies never\nbeat the simplest (two-link) worst case")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_poa_distributions(battery_results: dict, path: str) -> None:
    degrees = sorted(battery_results)
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [battery_results[p].poas for p in degrees]
    parts = ax.violinplot(data, positions=degrees, widths=0.6, showmedians=True)
    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_alpha(0.55)

    for p, color in zip(degrees, COLORS):
        ax.axhline(poa_bound(p), color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        ax.text(p + 0.28, poa_bound(p), rf"$\beta({p})$={poa_bound(p):.3f}", fontsize=8,
                va="center", color=color)

    ax.set_xticks(degrees)
    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel("empirical price of anarchy")
    ax.set_title("Empirical PoA across random series-parallel topologies\n(dashed lines: theoretical worst case for that degree)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence_two_link(convergence: dict, path: str) -> None:
    fig, axes = plt.subplots(1, len(convergence), figsize=(4.2 * len(convergence), 4.2), sharey=False)
    if len(convergence) == 1:
        axes = [axes]
    for ax, (p, color) in zip(axes, zip(sorted(convergence), COLORS)):
        pts = convergence[p]
        b = [pt.b for pt in pts]
        poa = [pt.poa for pt in pts]
        ax.plot(b, poa, "-", color=color, linewidth=2)
        ax.axhline(poa_bound(p), color="#1a202c", linestyle="--", linewidth=1.2,
                   label=rf"$\beta({p})$={poa_bound(p):.3f}")
        ax.set_xlabel("$b$ (constant-edge latency)")
        ax.set_title(f"degree $p={p}$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PoA($b$) on the two-link network")
    fig.suptitle("Sweeping the simplest possible network reproduces the bound exactly\n(generic NLP solver, not the closed-form shortcut)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_braess_example(cost_no_shortcut: float, cost_with_shortcut: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["without shortcut\n(2 routes)", "with shortcut\n(free A->B edge)"],
                   [cost_no_shortcut, cost_with_shortcut],
                   color=["#2f855a", "#c53030"])
    for bar, val in zip(bars, [cost_no_shortcut, cost_with_shortcut]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=11)
    ax.set_ylabel("equilibrium total travel cost")
    ax.set_title("Braess's paradox (solver sanity check):\nadding a free edge makes everyone worse off")
    ax.set_ylim(0, max(cost_no_shortcut, cost_with_shortcut) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
