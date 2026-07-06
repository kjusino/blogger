"""Figure generation for the grid-sweep results. All four figures required
by the project write-up are produced here from (a) the results table and
(b) one retrained "representative" surrogate used only for the attractor
visualization (figure 1).

Uses the non-interactive Agg backend so this runs headlessly in CI/CLI
contexts with no display.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _unique_sorted(values):
    return sorted(set(values))


def plot_attractor_true_vs_surrogate(true_traj: np.ndarray, sur_traj: np.ndarray,
                                      out_path: str, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    axes[0].plot(true_traj[:, 0], true_traj[:, 2], lw=0.3, color="tab:blue",
                 alpha=0.8)
    axes[0].set_title("True Lorenz-63 attractor")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")

    axes[1].plot(sur_traj[:, 0], sur_traj[:, 2], lw=0.3, color="tab:red",
                 alpha=0.8)
    axes[1].set_title("Surrogate-iterated trajectory")
    axes[1].set_xlabel("x")

    for ax in axes:
        ax.set_xlim(-25, 25)
        ax.set_ylim(0, 55)

    fig.suptitle("True vs. learned-surrogate long-horizon trajectory (x-z projection)"
                 f"{title_suffix}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lyapunov_error_vs_trainsize(rows: List[dict], out_path: str) -> None:
    noise_levels = _unique_sorted(r["noise_level"] for r in rows)
    train_sizes = _unique_sorted(r["train_size"] for r in rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(noise_levels)))

    for noise, color in zip(noise_levels, colors):
        means = []
        for ts in train_sizes:
            vals = [r["lambda1_abs_error"] for r in rows
                    if r["noise_level"] == noise and r["train_size"] == ts
                    and np.isfinite(r["lambda1_abs_error"])]
            means.append(np.mean(vals) if vals else np.nan)
        ax.plot(train_sizes, means, marker="o", color=color,
                label=f"noise={noise:g}")

    ax.set_xscale("log")
    ax.set_xlabel("training-set size (log scale)")
    ax.set_ylabel(r"mean $|\lambda_1^{surrogate} - \lambda_1^{true}|$")
    ax.set_title("Leading Lyapunov exponent error vs. training-set size")
    ax.legend(title="observation noise")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_chaos_detection_heatmap(rows: List[dict], out_path: str) -> None:
    noise_levels = _unique_sorted(r["noise_level"] for r in rows)
    hidden_widths = _unique_sorted(r["hidden_width"] for r in rows)

    grid = np.full((len(noise_levels), len(hidden_widths)), np.nan)
    for i, noise in enumerate(noise_levels):
        for j, width in enumerate(hidden_widths):
            vals = [1.0 if r["chaos_detected_correct"] else 0.0 for r in rows
                    if r["noise_level"] == noise and r["hidden_width"] == width]
            if vals:
                grid[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(hidden_widths)))
    ax.set_xticklabels(hidden_widths)
    ax.set_yticks(range(len(noise_levels)))
    ax.set_yticklabels([f"{n:g}" for n in noise_levels])
    ax.set_xlabel("hidden width")
    ax.set_ylabel("observation noise")
    ax.set_title("Chaos-detection accuracy: fraction with correct sign($\\lambda_1$)\n"
                 "(aggregated over train sizes & seeds)")

    for i in range(len(noise_levels)):
        for j in range(len(hidden_widths)):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black", fontsize=10)

    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mse_vs_lyapunov_error(rows: List[dict], out_path: str) -> float:
    """Scatter of one-step validation MSE vs |lambda1 error| across all grid
    configs. Returns the Pearson correlation coefficient (computed on
    log10(MSE) vs |error|, in plain numpy) -- the headline number for
    whether low one-step loss predicts preserved chaos.
    """
    mse = np.array([r["val_mse"] for r in rows])
    err = np.array([r["lambda1_abs_error"] for r in rows])
    finite = np.isfinite(mse) & np.isfinite(err) & (mse > 0)
    mse, err = mse[finite], err[finite]
    log_mse = np.log10(mse)

    corr_matrix = np.corrcoef(log_mse, err)
    pearson_r = float(corr_matrix[0, 1])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(mse, err, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.set_xscale("log")
    # A few configs (usually high-capacity/noisy ones) produce surrogates
    # whose iterated map blows up, giving a wildly large |lambda1 error|;
    # symlog keeps both the near-zero-error bulk of points and those
    # outliers legible in a single honest plot (data is not clipped).
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("one-step validation MSE (log scale)")
    ax.set_ylabel(r"$|\lambda_1^{surrogate} - \lambda_1^{true}|$ (symlog scale)")
    ax.set_title("Does low one-step loss predict preserved chaos?\n"
                 f"Pearson r(log10(MSE), |err|) = {pearson_r:.3f}  (n={len(mse)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pearson_r


def generate_all_figures(rows: List[dict], figures_dir: str,
                          representative=None) -> dict:
    """Generate all four figures. `representative` is an optional
    (row, net, mean, std, sur_traj, true_long_traj) tuple from
    experiment.train_representative_surrogate used for figure 1; if not
    provided, figure 1 is skipped.

    Returns a dict of summary numbers computed while plotting (currently
    just the Pearson correlation from the MSE-vs-error scatter).
    """
    os.makedirs(figures_dir, exist_ok=True)
    summary = {}

    if representative is not None:
        rep_row, net, mean, std, sur_traj, true_long_traj = representative
        plot_attractor_true_vs_surrogate(
            true_long_traj, sur_traj,
            os.path.join(figures_dir, "attractor_true_vs_surrogate.png"),
            title_suffix=(f" (train_size={rep_row['train_size']}, "
                          f"noise={rep_row['noise_level']:g}, "
                          f"width={rep_row['hidden_width']})"),
        )

    plot_lyapunov_error_vs_trainsize(
        rows, os.path.join(figures_dir, "lyapunov_error_vs_trainsize.png"))

    plot_chaos_detection_heatmap(
        rows, os.path.join(figures_dir, "chaos_detection_heatmap.png"))

    summary["pearson_r_logmse_vs_lambda1_error"] = plot_mse_vs_lyapunov_error(
        rows, os.path.join(figures_dir, "mse_vs_lyapunov_error_scatter.png"))

    return summary
