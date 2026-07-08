"""All matplotlib figures for the sweep results."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import EPSILONS, TrialResult, CellSummary
from .theory import alon_boppana_bound

DEGREE_COLORS = {3: "#4C72B0", 4: "#DD8452", 6: "#55A868", 10: "#C44E52"}
DPI = 150


def _cells_for_d(cells: list[CellSummary], d: int) -> list[CellSummary]:
    return sorted([c for c in cells if c.d == d], key=lambda c: c.n)


def plot_lambda2_vs_n(cells: list[CellSummary], degrees, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for d in degrees:
        dc = _cells_for_d(cells, d)
        ns = np.array([c.n for c in dc])
        means = np.array([c.lambda2_abs_mean for c in dc])
        stds = np.array([c.lambda2_abs_std for c in dc])
        color = DEGREE_COLORS.get(d, None)
        ax.errorbar(
            ns, means, yerr=stds, marker="o", capsize=3, color=color, label=f"d={d} (empirical)"
        )
        bound = alon_boppana_bound(d)
        ax.axhline(bound, color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax.text(ns[-1] * 1.05, bound, f"2√{d-1}≈{bound:.3f}", color=color, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xlabel("n (number of vertices)")
    ax.set_ylabel(r"$\lambda(G) = \max_{i\geq2}|\lambda_i|$")
    ax.set_title("Mean $\\lambda(G)$ vs n, with the Alon-Boppana bound $2\\sqrt{d-1}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_gap_loglog(cells: list[CellSummary], fits: dict, degrees, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for d in degrees:
        dc = _cells_for_d(cells, d)
        ns = np.array([c.n for c in dc])
        gaps = np.abs(np.array([c.gap_mean for c in dc]))
        mask = gaps > 0
        color = DEGREE_COLORS.get(d, None)
        ax.plot(ns[mask], gaps[mask], marker="o", color=color, label=f"d={d}")
        fit = fits.get(d, {})
        if fit.get("alpha") is not None:
            alpha, intercept = fit["alpha"], fit["intercept"]
            fitted = np.exp(intercept) * ns[mask] ** alpha
            ax.plot(
                ns[mask],
                fitted,
                color=color,
                linestyle=":",
                linewidth=1,
                label=f"  fit: gap $\\propto n^{{{alpha:.2f}}}$ ($R^2$={fit['r_squared']:.3f})",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (number of vertices)")
    ax.set_ylabel(r"$|\mathrm{mean}[\lambda(G)] - 2\sqrt{d-1}|$   (empirically, mean $\lambda(G)$ sits below the bound)")
    ax.set_title("Convergence of $\\lambda(G)$ to the Alon-Boppana bound (log-log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_within_eps_fraction(cells: list[CellSummary], degrees, out_path: str) -> None:
    """Two-sided concentration: fraction of graphs with |lambda(G) -
    2*sqrt(d-1)| <= eps, vs n. Empirically mean lambda(G) approaches the
    bound from below (see gap_convergence_loglog.png), so the one-sided
    "near-Ramanujan" check lambda(G) <= bound+eps is already close to 1
    across this entire n range (reported in summary.json) -- the two-sided
    version plotted here is the metric that actually shows the interesting
    n-dependence: concentration around the bound, from both sides."""
    fig, axes = plt.subplots(1, len(degrees), figsize=(4 * len(degrees), 4.5), sharey=True)
    if len(degrees) == 1:
        axes = [axes]
    for ax, d in zip(axes, degrees):
        dc = _cells_for_d(cells, d)
        ns = np.array([c.n for c in dc])
        for eps in EPSILONS:
            fracs = np.array([c.frac_close_to_bound[f"eps_{eps}"] for c in dc])
            ax.plot(ns, fracs, marker="o", label=f"$\\varepsilon$={eps}")
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"d={d}")
        ax.set_xlabel("n")
    axes[0].set_ylabel(r"fraction with $|\lambda(G) - 2\sqrt{d-1}| \leq \varepsilon$")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Concentration of $\\lambda(G)$ around the Alon-Boppana bound vs n")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_distribution_comparison(results: list[TrialResult], degrees, out_path: str) -> None:
    fig, axes = plt.subplots(1, len(degrees), figsize=(4 * len(degrees), 4.5))
    if len(degrees) == 1:
        axes = [axes]
    for ax, d in zip(axes, degrees):
        d_results = [r for r in results if r.d == d]
        ns_available = sorted(set(r.n for r in d_results))
        n_small, n_large = ns_available[0], ns_available[-1]
        small_vals = [r.lambda2_abs for r in d_results if r.n == n_small]
        large_vals = [r.lambda2_abs for r in d_results if r.n == n_large]
        bound = alon_boppana_bound(d)
        ax.hist(small_vals, bins=12, alpha=0.5, label=f"n={n_small}", color="#999999", density=True)
        ax.hist(
            large_vals,
            bins=12,
            alpha=0.6,
            label=f"n={n_large}",
            color=DEGREE_COLORS.get(d, "#333333"),
            density=True,
        )
        ax.axvline(bound, color="black", linestyle="--", linewidth=1, label=r"$2\sqrt{d-1}$")
        ax.set_title(f"d={d}")
        ax.set_xlabel(r"$\lambda(G)$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.suptitle("$\\lambda(G)$ distribution concentrates near the bound as n grows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_exact_validation(validation_rows: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["name"] for row in validation_rows]
    x = np.arange(len(labels))
    width = 0.35

    computed_l1 = [row["computed_lambda1"] for row in validation_rows]
    exact_l1 = [row["exact_lambda1"] for row in validation_rows]
    computed_l2abs = [row["computed_lambda2_abs"] for row in validation_rows]
    exact_l2abs = [row["exact_lambda2_abs"] for row in validation_rows]

    ax.bar(x - width / 2, exact_l1, width, label="exact $\\lambda_1$", color="#4C72B0", alpha=0.5)
    ax.scatter(x - width / 2, computed_l1, color="#4C72B0", marker="x", s=80, label="computed $\\lambda_1$", zorder=3)
    ax.bar(x + width / 2, exact_l2abs, width, label="exact $\\lambda(G)$", color="#C44E52", alpha=0.5)
    ax.scatter(
        x + width / 2, computed_l2abs, color="#C44E52", marker="x", s=80, label="computed $\\lambda(G)$", zorder=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("eigenvalue")
    ax.set_title("Eigensolver validation against hand-known exact spectra")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
