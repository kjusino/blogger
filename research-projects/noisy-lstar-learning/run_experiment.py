#!/usr/bin/env python3
"""Run the noisy-L* redundancy experiment grid end-to-end: simulate, write
results/grid_results.csv, and render the figures used in the README.

Usage:
    python3 run_experiment.py            # full grid (~a few minutes)
    python3 run_experiment.py --quick     # small smoke-test grid (seconds)
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiment import run_grid, DELTA_Q, STRATEGIES
from src.analysis import (
    success_rate_table,
    mean_field_table,
    repetition_factor_table,
    union_bound_success_prediction,
    sorted_noise_rates,
    theoretical_repetitions_curve,
)

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

STRATEGY_LABELS = {
    "none": "no redundancy (k=1)",
    "fixed5": "fixed redundancy (k=5)",
    "adaptive": "adaptive redundancy (Hoeffding k(p))",
}
STRATEGY_COLORS = {
    "none": "#d62728",
    "fixed5": "#ff7f0e",
    "adaptive": "#2ca02c",
}


def write_csv(results, path):
    fieldnames = [
        "target_id", "target_num_states", "noise_rate", "strategy", "seed",
        "theoretical_k", "success", "hypothesis_num_states", "equivalence_queries",
        "converged", "distinct_queries", "raw_queries",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


def plot_success_rate_vs_noise(results, path):
    noise_rates = sorted_noise_rates(results)
    table = success_rate_table(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy in STRATEGIES:
        ys = [table[(p, strategy)] * 100 for p in noise_rates]
        ax.plot(noise_rates, ys, marker="o", label=STRATEGY_LABELS[strategy],
                color=STRATEGY_COLORS[strategy])
    ax.set_xlabel("membership-oracle noise rate p")
    ax.set_ylabel("exact learning success rate (%)")
    ax.set_title("L* learning success vs. membership-query noise")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_query_overhead_vs_noise(results, path):
    noise_rates = sorted_noise_rates(results)
    rep_table = repetition_factor_table(results)
    theory = theoretical_repetitions_curve(noise_rates, DELTA_Q)
    fig, ax = plt.subplots(figsize=(7, 5))
    ys_empirical = [rep_table[(p, "adaptive")] for p in noise_rates]
    ys_theory = [theory[p] for p in noise_rates]
    ax.plot(noise_rates, ys_empirical, marker="o", label="empirical mean repetitions/query",
            color=STRATEGY_COLORS["adaptive"])
    ax.plot(noise_rates, ys_theory, linestyle="--", marker="x", color="black",
            label=r"Hoeffding-bound prediction $k(p,\delta_q)$")
    ax.set_xlabel("membership-oracle noise rate p")
    ax.set_ylabel("mean sub-queries per distinct membership query")
    ax.set_title("Query redundancy overhead: empirical vs. theoretical (adaptive strategy)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_success_vs_union_bound(results, path):
    noise_rates = sorted_noise_rates(results)
    success = success_rate_table(results)
    predicted = union_bound_success_prediction(results, "adaptive", DELTA_Q)
    fig, ax = plt.subplots(figsize=(7, 5))
    ys_obs = [success[(p, "adaptive")] * 100 for p in noise_rates]
    ys_pred = [predicted[(p,)] * 100 for p in noise_rates]
    ax.plot(noise_rates, ys_obs, marker="o", label="observed success rate (adaptive)",
            color=STRATEGY_COLORS["adaptive"])
    ax.plot(noise_rates, ys_pred, linestyle="--", marker="x", color="gray",
            label=r"union-bound lower bound $1 - \bar{Q}\cdot\delta_q$")
    ax.set_xlabel("membership-oracle noise rate p")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Observed success rate vs. union-bound prediction (adaptive strategy)")
    ax.set_ylim(80, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_target_heatmap(results, path):
    import numpy as np

    noise_rates = sorted_noise_rates(results)
    target_ids = sorted({r["target_id"] for r in results})
    target_sizes = {r["target_id"]: r["target_num_states"] for r in results}
    table = success_rate_table(results, by=("target_id", "noise_rate"))

    matrix = np.array([
        [table.get((tid, p), float("nan")) for p in noise_rates]
        for tid in target_ids
    ])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(matrix * 100, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(noise_rates)))
    ax.set_xticklabels([f"{p:.2f}" for p in noise_rates])
    ax.set_yticks(range(len(target_ids)))
    ax.set_yticklabels([f"{target_sizes[tid]} states" for tid in target_ids])
    ax.set_xlabel("noise rate p")
    ax.set_ylabel("target DFA")
    ax.set_title("Adaptive-redundancy success rate (%) by target complexity")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j] * 100:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="success rate (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke-test grid")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    if args.quick:
        noise_rates = [0.0, 0.2, 0.4]
        sizes = (5, 8)
        num_seeds = 3
        adaptive_only_for_heatmap = ("adaptive",)
    else:
        noise_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        sizes = (5, 7, 10, 13)
        num_seeds = 10
        adaptive_only_for_heatmap = ("adaptive",)

    print(f"Running grid: noise_rates={noise_rates}, sizes={sizes}, "
          f"strategies={STRATEGIES}, seeds/config={num_seeds}")
    results = run_grid(noise_rates=noise_rates, sizes=sizes, num_seeds=num_seeds)
    print(f"Completed {len(results)} runs.")

    csv_path = RESULTS_DIR / "grid_results.csv"
    write_csv(results, csv_path)
    print(f"Wrote {csv_path}")

    plot_success_rate_vs_noise(results, FIGURES_DIR / "success_rate_vs_noise.png")
    plot_query_overhead_vs_noise(results, FIGURES_DIR / "query_overhead_vs_noise.png")
    plot_success_vs_union_bound(results, FIGURES_DIR / "success_vs_union_bound.png")
    plot_per_target_heatmap(
        [r for r in results if r["strategy"] == "adaptive"],
        FIGURES_DIR / "per_target_success_heatmap.png",
    )
    print(f"Wrote figures to {FIGURES_DIR}")

    table = success_rate_table(results)
    print("\nSuccess rate by (noise_rate, strategy):")
    for p in sorted_noise_rates(results):
        row = " | ".join(f"{s}={table[(p, s)]*100:5.1f}%" for s in STRATEGIES)
        print(f"  p={p:.2f}: {row}")


if __name__ == "__main__":
    main()
