"""Plotting for the barren-plateau sweep."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import theory

COST_COLORS = {"global": "#d62728", "local": "#1f77b4"}
DEPTH_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _group(results, cost_type, depth):
    rows = [r for r in results if r.cost_type == cost_type and r.depth == depth]
    rows.sort(key=lambda r: r.n)
    return rows


def plot_variance_vs_n(results, depths, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cost_type in zip(axes, ["global", "local"]):
        for i, depth in enumerate(depths):
            rows = _group(results, cost_type, depth)
            if not rows:
                continue
            ns = np.array([r.n for r in rows])
            var = np.array([r.variance for r in rows])
            err = np.array([r.variance_stderr for r in rows])
            ax.errorbar(
                ns,
                var,
                yerr=err,
                marker=DEPTH_MARKERS[i % len(DEPTH_MARKERS)],
                label=f"depth={depth}",
                capsize=3,
            )
        ax.set_yscale("log", base=2)
        ax.set_xlabel("number of qubits n")
        ax.set_title(f"{cost_type} cost")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Var[$\partial C / \partial \theta_1$]  (log$_2$ scale)")
    fig.suptitle("Gradient variance vs. circuit width, by cost function and depth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit_slopes(fit_table, out_path):
    """fit_table: list of dicts with cost_type, depth, exp_slope, exp_r2, pow_r2."""
    depths = sorted({row["depth"] for row in fit_table})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for cost_type, color in COST_COLORS.items():
        ys = [
            next(r["exp_slope"] for r in fit_table if r["cost_type"] == cost_type and r["depth"] == d)
            for d in depths
        ]
        ax.plot(depths, [-y for y in ys], marker="o", color=color, label=f"{cost_type} cost")
    ax.set_xlabel("circuit depth (layers)")
    ax.set_ylabel(r"fitted exponential rate $b$  (Var $\sim 2^{-bn}$)")
    ax.set_xscale("log", base=2)
    ax.set_title("Barren-plateau steepness vs. depth")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    width = 0.35
    x = np.arange(len(depths))
    for offset, cost_type in zip([-width / 2, width / 2], ["global", "local"]):
        r2s = [
            next(r["exp_r2"] for r in fit_table if r["cost_type"] == cost_type and r["depth"] == d)
            for d in depths
        ]
        ax.bar(x + offset, r2s, width=width, color=COST_COLORS[cost_type], label=f"{cost_type} cost")
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.set_xlabel("circuit depth (layers)")
    ax.set_ylabel(r"exponential-model $R^2$")
    ax.set_ylim(0, 1.05)
    ax.set_title("Goodness of exponential fit vs. depth")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, depth, out_path):
    """For one representative depth, overlay data + both fitted models for both cost types."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for cost_type, color in COST_COLORS.items():
        rows = _group(results, cost_type, depth)
        ns = np.array([r.n for r in rows])
        var = np.array([r.variance for r in rows])
        ax.scatter(ns, var, color=color, label=f"{cost_type} cost (data)", zorder=3)

        exp_fit = theory.fit_exponential(ns, var)
        n_line = np.linspace(ns.min(), ns.max(), 100)
        ax.plot(
            n_line,
            2 ** (exp_fit.slope * n_line + exp_fit.intercept),
            color=color,
            linestyle="--",
            label=f"{cost_type} exp fit ($R^2$={exp_fit.r2:.3f})",
        )
    ax.set_yscale("log", base=2)
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel(r"Var[$\partial C/\partial\theta_1$]")
    ax.set_title(f"Local vs. global cost, depth={depth}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
