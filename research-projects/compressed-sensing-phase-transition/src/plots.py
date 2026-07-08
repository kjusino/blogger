"""Figure generation for the compressed-sensing phase-transition experiment.
Uses the non-interactive Agg backend so it runs headless."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .experiment import ExperimentResult, empirical_threshold
from .theory import phase_transition_curve


def plot_heatmaps(result: ExperimentResult, out_path: str) -> None:
    """One empirical success-probability heatmap per n, each with the
    theoretical phase-transition curve overlaid in white."""
    n_values = result.n_values
    fig, axes = plt.subplots(1, len(n_values), figsize=(5 * len(n_values), 4.5),
                              squeeze=False)
    theory_rho = np.linspace(result.rho_grid.min(), result.rho_grid.max(), 200)
    theory_delta = phase_transition_curve(theory_rho)

    for ax, n in zip(axes[0], n_values):
        mat = result.success_matrix(n)
        im = ax.imshow(mat, origin="lower", aspect="auto", vmin=0, vmax=1,
                        cmap="viridis",
                        extent=[result.delta_grid.min(), result.delta_grid.max(),
                                result.rho_grid.min(), result.rho_grid.max()])
        ax.plot(theory_delta, theory_rho, color="white", linewidth=2,
                label="ALMT theory")
        ax.set_title(f"n = {n}")
        ax.set_xlabel(r"$\delta = m/n$")
        ax.set_ylabel(r"$\rho = k/n$")
        ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(im, ax=axes[0].tolist(), label="empirical recovery probability")
    fig.suptitle("Basis-pursuit recovery: empirical phase transition vs. ALMT theory")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_curves(result: ExperimentResult, out_path: str) -> None:
    """Empirical 50%-crossing threshold curve per n, overlaid on the
    theoretical curve, to show convergence as n grows."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    theory_rho = np.linspace(1e-3, result.rho_grid.max(), 200)
    theory_delta = phase_transition_curve(theory_rho)
    ax.plot(theory_rho, theory_delta, "k--", linewidth=2, label="ALMT theory")

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(result.n_values)))
    for n, color in zip(result.n_values, colors):
        mat = result.success_matrix(n)
        thresholds = [empirical_threshold(result.delta_grid, mat[i])
                      for i in range(len(result.rho_grid))]
        rhos = [r for r, t in zip(result.rho_grid, thresholds) if t is not None]
        deltas = [t for t in thresholds if t is not None]
        ax.plot(rhos, deltas, marker="o", color=color, label=f"empirical, n={n}")

    ax.set_xlabel(r"$\rho = k/n$")
    ax.set_ylabel(r"critical $\delta = m/n$")
    ax.set_title("Empirical 50% recovery threshold vs. theory")
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transition_width(n_values: list[int], widths: list[float], out_path: str) -> None:
    """Transition-band width (delta at 90% success minus delta at 10%
    success, averaged over rho) vs. n, showing the sharpening predicted by
    concentration of measure as n -> infinity."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(n_values, widths, marker="o", color="crimson")
    ax.set_xlabel("n")
    ax.set_ylabel(r"mean transition width ($\delta_{90\%} - \delta_{10\%}$)")
    ax.set_title("Phase-transition sharpening as n grows")
    ax.set_xscale("log")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
