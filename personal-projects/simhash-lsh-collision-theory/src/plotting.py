"""Matplotlib figures for the three experiments. No seaborn dependency."""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import theory


def plot_single_hash_collision(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    thetas_theory = np.linspace(0, math.pi, 200)
    ax.plot(
        thetas_theory,
        [theory.single_hash_collision_prob(t) for t in thetas_theory],
        "k--",
        label="theory: 1 - theta/pi",
        linewidth=1.5,
    )
    for dim, group in df.groupby("dim"):
        group = group.sort_values("theta")
        ax.errorbar(
            group["theta"],
            group["empirical_prob"],
            yerr=group["stderr"],
            marker="o",
            markersize=4,
            linestyle="none",
            capsize=2,
            label=f"empirical, dim={dim}",
        )
    ax.set_xlabel("angle theta (radians)")
    ax.set_ylabel("Pr[single hyperplane hash agrees]")
    ax.set_title("Single random-hyperplane collision probability: empirical vs. theory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_and_or_threshold(
    df: pd.DataFrame, k: int, L: int, empirical_threshold: float, theory_threshold: float, out_path: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    thetas_theory = np.linspace(0, math.pi, 300)
    ax.plot(
        thetas_theory,
        [theory.or_of_bands_prob(t, k, L) for t in thetas_theory],
        "k--",
        label=f"theory: 1-(1-p^{k})^{L}",
    )
    df_sorted = df.sort_values("theta")
    ax.errorbar(
        df_sorted["theta"],
        df_sorted["empirical_recall"],
        yerr=df_sorted["stderr"],
        marker="o",
        markersize=4,
        linestyle="none",
        capsize=2,
        color="tab:blue",
        label="empirical recall",
    )
    ax.axhline(0.5, color="gray", linewidth=0.7, linestyle=":")
    ax.axvline(theory_threshold, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(empirical_threshold, color="tab:blue", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("angle theta (radians)")
    ax.set_ylabel("Pr[collide in >= 1 of L tables]")
    ax.set_title(
        f"AND-OR LSH S-curve (k={k}, L={L}): theory* theta={theory_threshold:.3f}, "
        f"empirical theta={empirical_threshold:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, rho_hat: float, rho_theory: float, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    n_vals = df["n"].to_numpy()
    y_vals = df["mean_candidates"].to_numpy()
    ax.errorbar(
        n_vals,
        y_vals,
        yerr=df["std_candidates"],
        marker="o",
        linestyle="none",
        capsize=2,
        color="tab:blue",
        label="empirical mean candidate-set size",
    )
    mask = y_vals > 0
    if mask.sum() >= 2:
        log_n = np.log(n_vals[mask])
        c = np.exp(np.polyfit(log_n, np.log(y_vals[mask]), 1)[1])
        n_grid = np.linspace(n_vals.min(), n_vals.max(), 100)
        ax.plot(n_grid, c * n_grid**rho_hat, "b-", alpha=0.6, label=f"fit: n^{rho_hat:.3f}")
        ax.plot(
            n_grid,
            c * n_grid**rho_theory,
            "k--",
            alpha=0.8,
            label=f"theory: n^{rho_theory:.3f}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("background dataset size n (log scale)")
    ax.set_ylabel("mean LSH candidate-set size (log scale)")
    ax.set_title("Sublinear query-cost scaling: fitted exponent vs. Indyk-Motwani rho")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
