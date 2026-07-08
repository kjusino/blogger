#!/usr/bin/env python3
"""Reproducible experiment sweep for the grokking-group-circuits project.

Runs to completion with no arguments in a few minutes and regenerates
`results/results.json` and every figure in `plots/`. See README.md for the
research question and the resulting findings.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grok.groups import CyclicGroup, DihedralGroup, QuaternionGroup
from grok.train import train_group_task, step_to_reach
from grok.spectral import basis_blocks, block_energies

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
PLOTS_DIR = os.path.join(ROOT, "plots")
RESULTS_DIR = os.path.join(ROOT, "results")

MAIN_GROUPS = [
    ("cyclic_59", lambda: CyclicGroup(59), dict(steps=8000, checkpoint_every=100)),
    ("dihedral_15", lambda: DihedralGroup(15), dict(steps=20000, checkpoint_every=200)),
    ("quaternion_8", lambda: QuaternionGroup(), dict(steps=20000, checkpoint_every=200)),
]
SEEDS = [0, 1, 2]
BASE_CFG = dict(emb_dim=32, hidden_dim=128, lr=1e-3, weight_decay=1.0,
                 train_frac=0.5, final_n_shuffles=300)

SWEEP_ORDERS = [8, 13, 23, 37, 47, 59]
SWEEP_CFG = dict(emb_dim=16, hidden_dim=64, lr=2e-3, weight_decay=1.0,
                  train_frac=0.5, steps=20000, checkpoint_every=200,
                  final_n_shuffles=50, seed=0)

# does giving Q8 more of its (already tiny) dataset let it grok after all,
# or is 64 total examples just not enough regardless of split?
Q8_TRAIN_FRAC_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]
Q8_FRAC_CFG = dict(emb_dim=16, hidden_dim=64, lr=2e-3, weight_decay=1.0,
                    steps=20000, checkpoint_every=200, final_n_shuffles=50, seed=0)


def run_main_sweep():
    runs = {}
    for group_name, group_factory, cfg in MAIN_GROUPS:
        runs[group_name] = []
        for seed in SEEDS:
            t0 = time.time()
            group = group_factory()
            result = train_group_task(group, seed=seed, **{**BASE_CFG, **cfg})
            elapsed = time.time() - t0
            h = result["history"]
            train95 = step_to_reach(h, "train_acc", 0.95)
            test95 = step_to_reach(h, "test_acc", 0.95)
            print(f"[{group_name} seed={seed}] {elapsed:.1f}s "
                  f"final_train={h['train_acc'][-1]:.3f} final_test={h['test_acc'][-1]:.3f} "
                  f"train95={train95} test95={test95} align_z={result['final_alignment']['z']:.2f}")
            grok_gap = (test95 / train95) if (train95 and test95) else None
            corr = float(np.corrcoef(h["concentration"], h["test_acc"])[0, 1])
            run_record = {
                "seed": seed,
                "order": group.order,
                "n_train": int(len(result["train_idx"])),
                "n_test": int(len(result["test_idx"])),
                "history": h,
                "train95_step": train95,
                "test95_step": test95,
                "grok_gap_ratio": grok_gap,
                "final_train_acc": h["train_acc"][-1],
                "final_test_acc": h["test_acc"][-1],
                "final_alignment": result["final_alignment"],
                "initial_concentration": result["initial_concentration"],
                "concentration_vs_test_acc_corr": corr,
                "elapsed_seconds": elapsed,
                "config": {**BASE_CFG, **cfg},
            }
            if seed == 0:
                run_record["params_we"] = result["params"]["W_E"]  # kept in-memory only
            runs[group_name].append(run_record)
    return runs


def run_quaternion_train_frac_sweep():
    points = []
    for train_frac in Q8_TRAIN_FRAC_SWEEP:
        t0 = time.time()
        result = train_group_task(QuaternionGroup(), train_frac=train_frac, **Q8_FRAC_CFG)
        elapsed = time.time() - t0
        h = result["history"]
        print(f"[Q8 train_frac={train_frac}] {elapsed:.1f}s n_train={len(result['train_idx'])} "
              f"final_test={h['test_acc'][-1]:.3f} z={result['final_alignment']['z']:.2f}")
        points.append({
            "train_frac": train_frac,
            "n_train": int(len(result["train_idx"])),
            "n_test": int(len(result["test_idx"])),
            "final_test_acc": h["test_acc"][-1],
            "final_alignment_z": result["final_alignment"]["z"],
            "elapsed_seconds": elapsed,
        })
    return points


def run_data_scale_sweep():
    points = []
    for order in SWEEP_ORDERS:
        t0 = time.time()
        group = CyclicGroup(order)
        result = train_group_task(group, **SWEEP_CFG)
        elapsed = time.time() - t0
        h = result["history"]
        test95 = step_to_reach(h, "test_acc", 0.95)
        print(f"[sweep C{order}] {elapsed:.1f}s n_train={len(result['train_idx'])} "
              f"final_test={h['test_acc'][-1]:.3f} test95={test95}")
        points.append({
            "order": order,
            "n_train": int(len(result["train_idx"])),
            "final_test_acc": h["test_acc"][-1],
            "test95_step": test95,
            "grokked": bool(h["test_acc"][-1] > 0.9),
            "final_alignment_z": result["final_alignment"]["z"],
            "elapsed_seconds": elapsed,
        })
    return points


def plot_training_curves(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (group_name, group_runs) in zip(axes, runs.items()):
        for r in group_runs:
            h = r["history"]
            ax.plot(h["step"], h["train_acc"], color="#4C72B0", alpha=0.35, lw=1)
            ax.plot(h["step"], h["test_acc"], color="#DD8452", alpha=0.35, lw=1)
        mean_train = np.mean([r["history"]["train_acc"] for r in group_runs], axis=0)
        mean_test = np.mean([r["history"]["test_acc"] for r in group_runs], axis=0)
        steps = group_runs[0]["history"]["step"]
        ax.plot(steps, mean_train, color="#4C72B0", lw=2.5, label="train acc (mean)")
        ax.plot(steps, mean_test, color="#DD8452", lw=2.5, label="test acc (mean)")
        ax.set_xscale("log")
        ax.set_title(group_name)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("Train vs. test accuracy over training (3 seeds each, thin lines; bold = mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_progress_measure(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (group_name, group_runs) in zip(axes, runs.items()):
        r = group_runs[0]
        h = r["history"]
        ax.plot(h["step"], h["test_acc"], color="#DD8452", lw=2, label="test acc")
        ax.plot(h["step"], h["concentration"], color="#55A868", lw=2, label="spectral concentration")
        ax.set_xscale("log")
        ax.set_title(f"{group_name} (seed 0)\ncorr(concentration, test acc) = {r['concentration_vs_test_acc_corr']:.2f}")
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("Does representation-theoretic structure emerge alongside generalization?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_alignment_bar(runs, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    names = list(runs.keys())
    means = [np.mean([r["final_alignment"]["z"] for r in runs[n]]) for n in names]
    stds = [np.std([r["final_alignment"]["z"] for r in runs[n]]) for n in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax.bar(names, means, yerr=stds, capsize=6, color=colors)
    ax.axhline(3.0, color="gray", linestyle="--", lw=1, label="z=3 significance threshold")
    ax.set_ylabel("final alignment z-score vs. row-shuffle null")
    ax.set_title("Spectral/representation-theoretic alignment at end of training")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_data_scale_threshold(sweep_points, runs, q8_frac_points, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = [p["n_train"] for p in sweep_points]
    ys = [p["final_test_acc"] for p in sweep_points]
    ax.plot(xs, ys, "o-", color="#4C72B0", label="cyclic groups (sweep)")
    for p in sweep_points:
        ax.annotate(f"C{p['order']}", (p["n_train"], p["final_test_acc"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    q8_xs = [p["n_train"] for p in q8_frac_points]
    q8_ys = [p["final_test_acc"] for p in q8_frac_points]
    ax.plot(q8_xs, q8_ys, "D--", color="#C44E52", alpha=0.7,
            label="Q8 (train_frac 0.5-0.9 sweep)")

    markers = {"cyclic_59": ("s", "#4C72B0"), "dihedral_15": ("^", "#55A868"),
               "quaternion_8": ("D", "#C44E52")}
    for name, group_runs in runs.items():
        mean_n_train = np.mean([r["n_train"] for r in group_runs])
        mean_acc = np.mean([r["final_test_acc"] for r in group_runs])
        marker, color = markers[name]
        ax.scatter([mean_n_train], [mean_acc], marker=marker, s=110, color=color,
                   edgecolor="black", zorder=5, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("# training examples")
    ax.set_ylabel("final test accuracy")
    ax.set_title("Grokking success vs. absolute training-set size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_embedding_spectrum(runs, group_factories, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (group_name, group_runs) in zip(axes, runs.items()):
        group = group_factories[group_name]()
        blocks = basis_blocks(group)
        embeddings = group_runs[0]["params_we"]
        energies = block_energies(embeddings, blocks)
        labels = [lbl for lbl, _ in blocks]
        ax.bar(range(len(labels)), energies / energies.sum(), color="#4C72B0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, fontsize=6)
        ax.set_title(group_name)
        ax.set_ylabel("fraction of embedding variance")
    fig.suptitle("Final embedding energy per irrep/frequency block (seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.time()

    print("=== Main 3-group comparison (3 seeds each) ===")
    runs = run_main_sweep()

    print("\n=== Data-scale sweep (cyclic groups, 1 seed each) ===")
    sweep_points = run_data_scale_sweep()

    print("\n=== Q8 train-fraction sweep (does more data let it grok?) ===")
    q8_frac_points = run_quaternion_train_frac_sweep()

    group_factories = {n: f for n, f, _ in MAIN_GROUPS}

    print("\n=== Generating plots ===")
    plot_training_curves(runs, os.path.join(PLOTS_DIR, "training_curves.png"))
    plot_progress_measure(runs, os.path.join(PLOTS_DIR, "progress_measure.png"))
    plot_alignment_bar(runs, os.path.join(PLOTS_DIR, "alignment_bar.png"))
    plot_data_scale_threshold(sweep_points, runs, q8_frac_points,
                               os.path.join(PLOTS_DIR, "data_scale_threshold.png"))
    plot_embedding_spectrum(runs, group_factories, os.path.join(PLOTS_DIR, "embedding_spectrum.png"))

    elapsed_total = time.time() - t_start
    print(f"\nTotal experiment runtime: {elapsed_total:.1f}s")

    # strip the raw embedding table before serializing (keep results.json light)
    for group_name in runs:
        del runs[group_name][0]["params_we"]

    results = {
        "main_sweep": runs,
        "data_scale_sweep": sweep_points,
        "quaternion_train_frac_sweep": q8_frac_points,
        "elapsed_seconds": elapsed_total,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {os.path.join(RESULTS_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
