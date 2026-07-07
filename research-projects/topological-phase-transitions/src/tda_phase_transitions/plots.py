"""Matplotlib visualizations for the phase-transition experiment."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import TrialResult, run_trial

MODEL_LABELS = {"er": "Erdos-Renyi", "rgg": "Random Geometric Graph", "chung_lu": "Chung-Lu (power law)"}
MODEL_COLORS = {"er": "#3b7dd8", "rgg": "#d87d3b", "chung_lu": "#3bd88f"}


def plot_model_curves(model: str, n: int, seed: int, out_path: str) -> TrialResult:
    result, curves = run_trial(model, n, seed=seed, return_curves=True)
    t = curves["thresholds"]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax0.plot(t, curves["beta0"], label=r"$\beta_0$ (components)", color="#444444")
    ax0.plot(t, curves["giant_frac"] * curves["beta0"].max(), "--", color="#999999",
             label="giant fraction (rescaled)")
    ax0.set_ylabel(r"$\beta_0$")
    ax0.set_title(f"{MODEL_LABELS[model]}, n={n}")
    ax0.legend(loc="upper right", fontsize=8)

    ax1b = ax1.twinx()
    ax1.plot(t, curves["beta1"], color=MODEL_COLORS[model], label=r"$\beta_1$ (cycles)")
    ax1b.plot(t, curves["susceptibility"], color="#888888", alpha=0.7, label="susceptibility")
    ax1.set_ylabel(r"$\beta_1$", color=MODEL_COLORS[model])
    ax1b.set_ylabel("susceptibility")
    ax1.set_xlabel("filtration threshold t")

    ax1.axvline(result.theory_threshold, color="black", linestyle=":", label="theory threshold")
    ax1.axvline(result.percolation_threshold, color="red", linestyle="--", label="percolation (susceptibility peak)")
    if result.cycle_onset_threshold is not None:
        ax1.axvline(result.cycle_onset_threshold, color="purple", linestyle="-.", label="cycle onset")
    ax1.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return result


def plot_threshold_convergence(results: List[TrialResult], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODEL_LABELS:
        model_results = [r for r in results if r.model == model]
        n_values = sorted(set(r.n for r in model_results))
        means, stds = [], []
        for n in n_values:
            ratios = np.array(
                [r.percolation_threshold / r.theory_threshold for r in model_results if r.n == n]
            )
            means.append(ratios.mean())
            stds.append(ratios.std())
        ax.errorbar(
            n_values, means, yerr=stds, marker="o", capsize=3,
            label=MODEL_LABELS[model], color=MODEL_COLORS[model],
        )
    ax.axhline(1.0, color="black", linestyle=":", label="theory = 1.0")
    ax.set_xlabel("n (number of nodes)")
    ax.set_ylabel("percolation threshold / theory threshold")
    ax.set_title("Finite-size convergence of the topological percolation detector")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cycle_vs_percolation(results: List[TrialResult], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    models = list(MODEL_LABELS)
    width = 0.35
    positions = np.arange(len(models))

    perc_means, perc_stds, onset_means, onset_stds = [], [], [], []
    for model in models:
        model_results = [r for r in results if r.model == model]
        perc = np.array([r.percolation_threshold / r.theory_threshold for r in model_results])
        onset = np.array(
            [r.cycle_onset_threshold / r.theory_threshold for r in model_results if r.cycle_onset_threshold is not None]
        )
        perc_means.append(perc.mean())
        perc_stds.append(perc.std())
        onset_means.append(onset.mean() if onset.size else 0.0)
        onset_stds.append(onset.std() if onset.size else 0.0)

    ax.bar(positions - width / 2, perc_means, width, yerr=perc_stds, capsize=3,
           label="percolation threshold (susceptibility peak)", color="#d84b3b")
    ax.bar(positions + width / 2, onset_means, width, yerr=onset_stds, capsize=3,
           label="cycle onset threshold ($H_1$)", color="#7d3bd8")
    ax.axhline(1.0, color="black", linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_ylabel("threshold / theory threshold")
    ax.set_title("Topological cycle onset vs. percolation threshold, by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_finite_size_collapse(model: str, n_values: List[int], seed_base: int, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, n in enumerate(n_values):
        result, curves = run_trial(model, n, seed=seed_base + i, grid_points=300, return_curves=True)
        rescaled_t = curves["thresholds"] / result.theory_threshold
        ax.plot(rescaled_t, curves["susceptibility"] / n, label=f"n={n}")
    ax.set_xlabel(r"$t / t_{theory}$ (rescaled threshold)")
    ax.set_ylabel(r"$\chi(t) / n$ (rescaled susceptibility)")
    ax.set_title(f"Finite-size scaling collapse: {MODEL_LABELS[model]}")
    ax.set_xlim(0, 3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
