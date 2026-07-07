"""Figure generation for the double-descent experiment. Pure functions of
already-computed results so plotting never re-runs simulations."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .theory import asymptotic_risk


def plot_risk_vs_gamma(results, r2, sigma2, out_path):
    gammas = np.array([r["gamma"] for r in results])
    emp = np.array([r["empirical_risk"] for r in results])
    err = np.array([r["empirical_stderr"] for r in results])
    theory = np.array([r["theoretical_risk"] for r in results])

    dense = np.linspace(0.05, max(gammas.max(), 5.0), 2000)
    dense = dense[np.abs(dense - 1.0) > 0.01]
    theory_dense = np.array([asymptotic_risk(g, r2, sigma2) for g in dense])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(dense, theory_dense, color="#1f77b4", lw=2,
            label="Asymptotic theory $R(\\gamma)$")
    ax.errorbar(gammas, emp, yerr=err, fmt="o", color="#d62728",
                ms=5, capsize=3, label="Empirical risk (Monte Carlo)",
                zorder=5)
    ax.axvline(1.0, color="gray", ls="--", lw=1,
               label="Interpolation threshold $\\gamma=1$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma = p / n$")
    ax.set_ylabel(r"Test risk $E\|\hat\beta - \beta_0\|^2$ (log scale)")
    ax.set_title("Double descent: empirical risk vs. closed-form theory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_relative_error(results, out_path, threshold):
    gammas = np.array([r["gamma"] for r in results])
    rel_err = np.array([r["relative_error"] for r in results]) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gammas, rel_err, "o-", color="#2ca02c")
    ax.axhline(threshold * 100, color="gray", ls="--",
               label=f"{threshold*100:.0f}% success threshold")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"$\gamma = p / n$")
    ax.set_ylabel("Relative error vs. exact theory (%)")
    ax.set_title("Empirical-vs-theoretical agreement across the sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bias_variance_decomposition(results, out_path):
    gammas = np.array([r["gamma"] for r in results])
    emp_bias2 = np.array([r["empirical_bias2"] for r in results])
    th_bias2 = np.array([r["theoretical_bias2"] for r in results])
    emp_var = np.array([r["empirical_variance"] for r in results])
    th_var = np.array([r["theoretical_variance"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(gammas, th_bias2, color="#9467bd", lw=2, label=r"Theory bias$^2$")
    ax.scatter(gammas, emp_bias2, color="#9467bd", marker="x", s=40,
               label=r"Empirical bias$^2$")
    ax.plot(gammas, th_var, color="#ff7f0e", lw=2, label="Theory variance")
    ax.scatter(gammas, emp_var, color="#ff7f0e", marker="o", s=25,
               facecolors="none", label="Empirical variance")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma = p / n$")
    ax.set_ylabel("Component of risk (log scale)")
    ax.set_title("Bias$^2$/variance decomposition: empirical vs. theory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ridge_sweep(ridge_results, out_path):
    lambdas = sorted(set(r["lambda"] for r in ridge_results))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    for i, lam in enumerate(lambdas):
        rows = [r for r in ridge_results if r["lambda"] == lam]
        rows.sort(key=lambda r: r["gamma"])
        gammas = [r["gamma"] for r in rows]
        emp = [r["empirical_risk"] for r in rows]
        err = [r["empirical_stderr"] for r in rows]
        color = cmap(i / max(len(lambdas) - 1, 1))
        ax.errorbar(gammas, emp, yerr=err, fmt="o-", color=color,
                    capsize=3, label=f"$\\lambda = {lam:g}$")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma = p / n$")
    ax.set_ylabel("Empirical test risk (log scale)")
    ax.set_title("Ridge regularization suppresses the interpolation peak")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
