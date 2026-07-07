"""Matplotlib figures for the hypercube-cutoff experiment."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_tv_curves_vs_t(curve_rows, n_subset, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in n_subset:
        rows = [r for r in curve_rows if r["n"] == n]
        rows.sort(key=lambda r: r["t"])
        t = [r["t"] for r in rows]
        tv = [r["exact_tv"] for r in rows]
        ax.plot(t, tv, marker="o", markersize=3, label=f"n={n}")
    ax.set_yscale("log")
    ax.set_xlabel("time t (steps)")
    ax.set_ylabel("exact TV distance to uniform (log scale)")
    ax.set_title("Exact mixing curves: a sharp cutoff, not gradual decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_data_collapse(curve_rows, n_subset, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in n_subset:
        rows = [r for r in curve_rows if r["n"] == n]
        rows.sort(key=lambda r: r["c"])
        c = np.array([r["c"] for r in rows])
        tv = np.array([r["exact_tv"] for r in rows])
        ax.plot(c, tv, marker="o", markersize=3, label=f"n={n}")

    c_fine = np.linspace(min(r["c"] for r in curve_rows), max(r["c"] for r in curve_rows), 300)
    from .theory import limiting_profile
    profile = np.minimum(limiting_profile(c_fine), 1.0)
    ax.plot(c_fine, profile, "k--", linewidth=2,
            label="chi-square (Diaconis-Shahshahani) bound profile")

    ax.set_xlabel("rescaled time c = (2t - n ln n) / n")
    ax.set_ylabel("exact TV distance")
    ax.set_title("Exact-TV curves collapse onto each other -- but not onto the chi-square bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cutoff_time_scaling(summary_rows, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    n_vals = np.array([r["n"] for r in summary_rows])
    t_half = np.array([r["t_half_empirical"] for r in summary_rows])
    t_star = np.array([r["t_star_theory"] for r in summary_rows])

    ax1.loglog(n_vals, t_half, "o-", label="empirical $t_{1/2}$ (exact TV = 0.5)")
    ax1.loglog(n_vals, t_star, "k--", label="theory $t^* = (n \\ln n)/2$")
    ax1.set_xlabel("n")
    ax1.set_ylabel("time")
    ax1.set_title("Half-mixing time vs. asymptotic cutoff location")
    ax1.legend()

    rel_err = np.array([r["rel_err_half_vs_cutoff_time"] for r in summary_rows])
    ax2.loglog(n_vals, rel_err, "o-", color="tab:red")
    ax2.set_xlabel("n")
    ax2.set_ylabel("relative error $|t_{1/2} - t^*| / t^*$")
    ax2.set_title("Relative error shrinks as n grows")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_window_scaling(summary_rows, out_path: str) -> None:
    n_vals = np.array([r["n"] for r in summary_rows], dtype=float)
    window = np.array([r["window"] for r in summary_rows], dtype=float)
    slope, intercept = np.polyfit(np.log(n_vals), np.log(window), 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(n_vals, window, "o", label="empirical window ($t_{0.25} - t_{0.75}$)")
    fit = np.exp(intercept) * n_vals ** slope
    ax.loglog(n_vals, fit, "--", label=f"fit: window $\\propto n^{{{slope:.2f}}}$")
    ax.set_xlabel("n")
    ax.set_ylabel("window")
    ax.set_title("Window scaling (theory predicts linear, slope = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_collapse_errors(summary_rows, out_path: str) -> None:
    n_vals = np.array([r["n"] for r in summary_rows], dtype=float)
    self_err = np.array([r["self_collapse_error"] for r in summary_rows], dtype=float)
    bound_gap = np.array([r["chi_square_bound_gap"] for r in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    # drop the last point of self_err (it's the reference curve, error = 0, undefined on log scale)
    mask = self_err > 0
    ax.loglog(n_vals[mask], self_err[mask], "o-",
              label="self-collapse error (vs. largest-n curve) -- shrinks with n")
    ax.loglog(n_vals, bound_gap, "s-", color="tab:red",
              label="gap vs. chi-square-bound profile -- stays $O(1)$")
    ax.set_xlabel("n")
    ax.set_ylabel("max |exact TV(c) - reference(c)|")
    ax.set_title("True TV curves collapse; the classical chi-square bound is not asymptotically tight")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mc_validation(mc_rows, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    c = np.array([r["c"] for r in mc_rows])
    exact = np.array([r["exact_tv"] for r in mc_rows])
    bv = np.array([r["mc_bitvector_tv"] for r in mc_rows])
    bv_lo = np.array([r["mc_bitvector_ci_lo"] for r in mc_rows])
    bv_hi = np.array([r["mc_bitvector_ci_hi"] for r in mc_rows])

    ax.plot(c, exact, "k-", linewidth=2, label="exact TV (birth-death chain)")
    ax.errorbar(c, bv, yerr=[bv - bv_lo, bv_hi - bv], fmt="o", capsize=3,
                label="Monte Carlo (literal bit-vector walkers, 95% CI)")
    ax.set_xlabel("rescaled time c")
    ax.set_ylabel("TV distance")
    ax.set_title(f"Monte Carlo simulation vs. exact chain (n={mc_rows[0]['n']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lumping_check(lumping_rows, out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    lumped = np.array([r["exact_lumped_tv"] for r in lumping_rows])
    full = np.array([r["exact_bruteforce_tv"] for r in lumping_rows])
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.scatter(lumped, full, s=25)
    ax1.set_xlabel("lumped birth-death chain TV")
    ax1.set_ylabel("brute-force 2^n chain TV")
    ax1.set_title("Lumping is exact: birth-death TV == full-chain TV")

    n_vals = sorted(set(r["n"] for r in lumping_rows))
    for n in n_vals:
        rows = [r for r in lumping_rows if r["n"] == n]
        rows.sort(key=lambda r: r["t"])
        t = [r["t"] for r in rows]
        exact = [r["exact_lumped_tv"] for r in rows]
        mc = [r["mc_bitvector_tv"] for r in rows]
        mc_lo = [r["mc_ci_lo"] for r in rows]
        mc_hi = [r["mc_ci_hi"] for r in rows]
        line = ax2.plot(t, exact, "-", label=f"n={n} exact")[0]
        ax2.errorbar(t, mc, yerr=[np.array(mc) - np.array(mc_lo), np.array(mc_hi) - np.array(mc)],
                     fmt="o", color=line.get_color(), capsize=2, markersize=4)
    ax2.set_xlabel("t")
    ax2.set_ylabel("TV distance")
    ax2.set_title("Literal bit-vector Monte Carlo vs. exact (small n)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
