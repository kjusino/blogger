import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from src.gue_theory import gue_surmise_pdf, poisson_pdf, montgomery_pair_correlation


def plot_spacing_histograms(height_results, path):
    """2x2 grid: empirical spacing histogram vs GUE surmise vs Poisson,
    one panel per height-sweep window."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    s_grid = np.linspace(0.001, 3.5, 400)
    for ax, res in zip(axes.flat, height_results):
        ax.hist(
            res["spacings"], bins=18, range=(0, 3.5), density=True,
            alpha=0.55, color="#4c72b0", label="empirical",
        )
        ax.plot(s_grid, gue_surmise_pdf(s_grid), color="#c44e52", lw=2, label="GUE surmise")
        ax.plot(s_grid, poisson_pdf(s_grid), color="#55a868", lw=2, ls="--", label="Poisson")
        ax.set_title(f"{res['label']}  (T≈{res['t_min']:.0f}-{res['t_max']:.0f}, N={res['n_spacings']})")
        ax.set_xlabel("unfolded spacing s")
        ax.set_ylabel("density")
    axes.flat[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Unfolded nearest-neighbor spacing of Riemann zeta zeros vs GUE / Poisson")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pair_correlation(height_results, path):
    """2x2 grid: empirical pair-correlation R2_hat vs Montgomery's
    conjectured form vs the flat (Poisson) reference, one panel per
    height-sweep window."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    u_grid = np.linspace(0.001, 3.0, 400)
    for ax, res in zip(axes.flat, height_results):
        ax.plot(res["bin_centers"], res["r2_hat"], "o-", color="#4c72b0", ms=4, label="empirical R2")
        ax.plot(u_grid, montgomery_pair_correlation(u_grid), color="#c44e52", lw=2, label="Montgomery")
        ax.axhline(1.0, color="#55a868", lw=2, ls="--", label="Poisson (R2=1)")
        ax.set_title(f"{res['label']}  (T≈{res['t_min']:.0f}-{res['t_max']:.0f}, N={res['n_spacings']})")
        ax.set_xlabel("unfolded separation u")
        ax.set_ylabel("R2(u)")
        ax.set_ylim(-0.05, 1.4)
    axes.flat[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Empirical pair correlation of zeta zeros vs Montgomery's conjecture")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ks_vs_height(height_results, path):
    """KS statistic (empirical spacings vs GUE surmise) across the height
    sweep, height (T, log scale) on the x-axis."""
    t_mid = [np.sqrt(r["t_min"] * max(r["t_max"], 1.0)) for r in height_results]
    ks_gue = [r["ks_gue"] for r in height_results]
    ks_poisson = [r["ks_poisson"] for r in height_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_mid, ks_gue, "o-", color="#c44e52", label="KS distance to GUE surmise")
    ax.plot(t_mid, ks_poisson, "s--", color="#55a868", label="KS distance to Poisson")
    ax.set_xscale("log")
    ax.set_xlabel("window height T (log scale)")
    ax.set_ylabel("KS statistic")
    ax.set_title("Fit quality vs height on the critical line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_scaling_vs_n(subsample_results, fit, path):
    """Log-log plot of KS statistic (vs GUE) against sample size N for
    the nested-subsample sweep, with the OLS fit line and a reference
    slope of -1/2 (the finite-sample-noise prediction)."""
    ns = np.array([r["n_spacings"] for r in subsample_results], dtype=float)
    ks = np.array([r["ks_gue"] for r in subsample_results], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ns, ks, "o", color="#4c72b0", ms=8, label="empirical KS(N)")

    x_line = np.linspace(ns.min(), ns.max(), 50)
    fit_line = np.exp(fit["intercept"]) * x_line ** fit["slope"]
    ax.loglog(x_line, fit_line, "-", color="#c44e52",
              label=f"OLS fit: slope={fit['slope']:.2f}")

    ref = ks[0] * (x_line / ns[0]) ** -0.5
    ax.loglog(x_line, ref, "--", color="#55a868", label="reference slope = -1/2")

    ax.set_xlabel("N (spacings in window)")
    ax.set_ylabel("KS statistic vs GUE surmise")
    ax.set_title("Does the GUE-fit gap shrink like finite-sample noise (N^-1/2)?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
