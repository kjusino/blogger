"""Figure generation from committed result CSVs (no fabricated numbers --
every plot reads its data from results/*.csv produced by run_experiment.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .patterns import arcsin_law

RHO_COLORS = {
    0.0: "#1b9e77",
    0.1: "#d95f02",
    0.2: "#7570b3",
    0.3: "#e7298a",
    0.4: "#66a61e",
    0.5: "#e6ab02",
}


def plot_overlap_vs_alpha(sweep_rows: list[dict], out_path: Path) -> None:
    ns = sorted({r["n"] for r in sweep_rows})
    fig, axes = plt.subplots(1, len(ns), figsize=(5 * len(ns), 4.2), sharey=True)
    if len(ns) == 1:
        axes = [axes]
    rhos = sorted({r["rho"] for r in sweep_rows})

    for ax, n in zip(axes, ns):
        for rho in rhos:
            pts = [r for r in sweep_rows if r["n"] == n and r["rho"] == rho]
            pts.sort(key=lambda r: r["alpha"])
            alphas = [r["alpha"] for r in pts]
            means = [r["mean_overlap"] for r in pts]
            sems = [r["sem_overlap"] for r in pts]
            color = RHO_COLORS.get(rho, None)
            ax.errorbar(alphas, means, yerr=sems, marker="o", ms=3, lw=1.2,
                        label=f"rho={rho}", color=color, capsize=2)
        ax.axhline(0.95, color="gray", ls="--", lw=0.8, label="_nolegend_")
        ax.set_title(f"N = {n}")
        ax.set_xlabel(r"$\alpha = P/N$")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("mean retrieval overlap |m|")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Retrieval overlap vs. load, by correlation rho")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_finite_size_scaling(critical_alphas: list[dict], extrapolation: list[dict], out_path: Path) -> None:
    rhos = sorted({r["rho"] for r in extrapolation})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for rho in rhos:
        pts = [r for r in critical_alphas if r["rho"] == rho and r["alpha_c"] is not None]
        pts.sort(key=lambda r: r["n"])
        if len(pts) < 2:
            continue
        inv_n = np.array([1.0 / r["n"] for r in pts])
        alpha_c = np.array([r["alpha_c"] for r in pts])
        color = RHO_COLORS.get(rho, None)
        ax.scatter(inv_n, alpha_c, color=color, label=f"rho={rho}", zorder=3)

        fit = next(e for e in extrapolation if e["rho"] == rho)
        x_line = np.array([0.0, inv_n.max() * 1.15])
        y_line = fit["alpha_c_inf"] + fit["slope"] * x_line
        ax.plot(x_line, y_line, color=color, lw=1, ls="--", zorder=2)
        ax.errorbar([0.0], [fit["alpha_c_inf"]], yerr=[fit["alpha_c_inf_stderr"]],
                    color=color, marker="*", ms=12, capsize=3, zorder=4)

    ax.set_xlabel("1/N")
    ax.set_ylabel(r"empirical $\alpha_c(N)$")
    ax.set_title("Finite-size scaling: alpha_c(N) vs 1/N\n(stars = linear extrapolation to N -> infinity)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_diagram(extrapolation: list[dict], fit: dict, out_path: Path, classical_alpha_c: float = 0.138) -> None:
    extrapolation = sorted(extrapolation, key=lambda r: r["rho"])
    rhos = np.array([r["rho"] for r in extrapolation])
    alpha_c_inf = np.array([r["alpha_c_inf"] for r in extrapolation])
    errs = np.array([r["alpha_c_inf_stderr"] for r in extrapolation])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.errorbar(rhos, alpha_c_inf, yerr=errs, fmt="o", ms=6, capsize=3,
                color="black", label="empirical alpha_c(inf, rho)", zorder=4)

    rho_dense = np.linspace(0, max(rhos), 200)
    alpha0 = fit["alpha0"]
    linear_curve = alpha0 * (1 - rho_dense)
    power_curve = alpha0 * np.power(1 - rho_dense, fit["power_k"])
    ax.plot(rho_dense, linear_curve, color="#d95f02", lw=1.5,
            label=f"H1 linear: alpha_c(0)*(1-rho)  [RMSE={fit['linear_rmse']:.4f}]")
    ax.plot(rho_dense, power_curve, color="#7570b3", lw=1.5, ls="--",
            label=f"power law: alpha_c(0)*(1-rho)^{fit['power_k']:.2f}  [RMSE={fit['power_rmse']:.4f}]")
    ax.axhline(classical_alpha_c, color="gray", ls=":", lw=1,
               label=f"AGS 1985 classical value ({classical_alpha_c})")

    ax.set_xlabel("rho (pattern correlation)")
    ax.set_ylabel(r"extrapolated $\alpha_c(\infty, \rho)$")
    ax.set_title("Phase diagram: extrapolated critical capacity vs. correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_arcsin_validation(arcsin_rows: list[dict], out_path: Path) -> None:
    rhos_input = np.linspace(0, max(r["rho"] for r in arcsin_rows), 200)
    theoretical_curve = arcsin_law(rhos_input)

    rows = sorted(arcsin_rows, key=lambda r: r["rho"])
    rhos = [r["rho"] for r in rows]
    emp = [r["empirical_corr_mean"] for r in rows]
    sem = [r["empirical_corr_sem"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rhos_input, theoretical_curve, color="gray", lw=1.5,
            label=r"theory: $(2/\pi)\arcsin(\rho)$")
    ax.errorbar(rhos, emp, yerr=sem, fmt="o", ms=6, capsize=3, color="#d95f02",
                label="empirical pairwise +-1 correlation")
    ax.set_xlabel("input Gaussian correlation rho")
    ax.set_ylabel("output +-1 correlation")
    ax.set_title("Arcsin-law validation of the correlated pattern generator")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
