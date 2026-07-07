"""Matplotlib figures for the BBP phase-transition sweep."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_eigenvalue_vs_lambda(results, c_target, out_path):
    rows = [r for r in results if np.isclose(r.c, c_target)]
    rows.sort(key=lambda r: r.lam)
    ratios = [r.lam_over_threshold for r in rows]
    mean_eig = [r.mean_eig for r in rows]
    ci = [r.ci95_eig for r in rows]
    theory = [r.theory_eig for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ratios, mean_eig, yerr=ci, fmt="o", label="empirical (95% CI)", color="tab:blue")
    ax.plot(ratios, theory, "-", label="BBP theory", color="tab:red")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="threshold lam*=sqrt(c)")
    ax.set_xlabel("lam / lam* (spike strength relative to BBP threshold)")
    ax.set_ylabel("top eigenvalue of sample covariance")
    ax.set_title(f"Top eigenvalue vs. spike strength (c={c_target:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alignment_vs_lambda(results, c_target, out_path):
    rows = [r for r in results if np.isclose(r.c, c_target)]
    rows.sort(key=lambda r: r.lam)
    ratios = [r.lam_over_threshold for r in rows]
    mean_align = [r.mean_align for r in rows]
    ci = [r.ci95_align for r in rows]
    theory = [r.theory_align for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ratios, mean_align, yerr=ci, fmt="o", label="empirical (95% CI)", color="tab:blue")
    ax.plot(ratios, theory, "-", label="BBP theory", color="tab:red")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="threshold lam*=sqrt(c)")
    ax.set_xlabel("lam / lam* (spike strength relative to BBP threshold)")
    ax.set_ylabel("squared alignment |<u_hat, v>|^2")
    ax.set_title(f"Eigenvector alignment vs. spike strength (c={c_target:g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_diagram(results, out_path):
    c_values = sorted(set(r.c for r in results))
    ratio_values = sorted(set(round(r.lam_over_threshold, 6) for r in results))
    grid = np.full((len(ratio_values), len(c_values)), np.nan)
    for r in results:
        i = ratio_values.index(round(r.lam_over_threshold, 6))
        j = c_values.index(r.c)
        grid[i, j] = r.mean_align

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[min(c_values), max(c_values), min(ratio_values), max(ratio_values)],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.axhline(1.0, color="white", linestyle="--", linewidth=1.5, label="lam=lam* (theory)")
    ax.set_xlabel("c = p / n")
    ax.set_ylabel("lam / lam*")
    ax.set_title("Empirical alignment |<u_hat, v>|^2 phase diagram")
    fig.colorbar(im, ax=ax, label="mean squared alignment")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_crossing(crossing_results, out_path):
    fig, axes = plt.subplots(1, len(crossing_results), figsize=(5 * len(crossing_results), 4.5), squeeze=False)
    for ax, (c, lam_hat, thr, lams, aligns) in zip(axes[0], crossing_results):
        ax.plot(lams, aligns, "o-", color="tab:blue", label="empirical mean alignment")
        ax.axhline(0.05, color="gray", linestyle=":", label="crossing=0.05")
        ax.axvline(thr, color="tab:red", linestyle="--", label=f"theory lam*={thr:.3f}")
        if lam_hat is not None:
            ax.axvline(lam_hat, color="tab:green", linestyle="--", label=f"empirical lam_hat={lam_hat:.3f}")
        ax.set_xlabel("lam")
        ax.set_ylabel("mean squared alignment")
        ax.set_title(f"c={c:g}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_finite_size_scaling(scaling_rows_by_regime, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, rows in scaling_rows_by_regime.items():
        rows = sorted(rows, key=lambda r: r["p"])
        p_vals = [r["p"] for r in rows]
        errs = [max(r["abs_err_eig"], 1e-12) for r in rows]
        ax.loglog(p_vals, errs, "o-", label=label)
    ax.set_xlabel("p (dimension)")
    ax.set_ylabel("|empirical top eigenvalue - theory|")
    ax.set_title("Finite-size convergence to the BBP asymptotic prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
