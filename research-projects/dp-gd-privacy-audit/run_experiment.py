#!/usr/bin/env python3
"""CLI entry point for the DP-GD privacy-audit experiment grid.

Usage:
    python3 run_experiment.py            # full grid, N=2000 trials/world
    python3 run_experiment.py --quick     # fast smoke test, N=200 trials/world
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.experiment import run_grid, save_results  # noqa: E402

RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"


def plot_theory_vs_audit(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sigmas = sorted({r["sigma"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(sigmas)))
    for sigma, color in zip(sigmas, colors):
        sub = sorted([r for r in rows if r["sigma"] == sigma], key=lambda r: r["T"])
        Ts = [r["T"] for r in sub]
        theory = [r["epsilon_theory"] for r in sub]
        audit = [r["eps_lower"] for r in sub]
        ax.plot(Ts, theory, "-", color=color, label=f"theory, sigma={sigma}")
        ax.plot(Ts, audit, "--", marker="o", color=color, label=f"audit (eps_lower), sigma={sigma}")
    ax.set_xlabel("T (composed steps)")
    ax.set_ylabel("epsilon (delta=1e-5)")
    ax.set_title("RDP-accountant theory vs. Monte Carlo audit lower bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ratio(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sigmas = sorted({r["sigma"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(sigmas)))
    for sigma, color in zip(sigmas, colors):
        sub = sorted([r for r in rows if r["sigma"] == sigma], key=lambda r: r["T"])
        Ts = [r["T"] for r in sub]
        ratio = [r["ratio_audit_over_theory"] for r in sub]
        ax.plot(Ts, ratio, "-o", color=color, label=f"sigma={sigma}")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("T (composed steps)")
    ax.set_ylabel("eps_lower / epsilon_theory")
    ax.set_title("Audit-to-theory ratio vs. composition depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(roc_records: dict, sigma_for_roc: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for T, style in [(1, "-o"), (8, "-s")]:
        roc = roc_records.get((sigma_for_roc, T))
        if roc is None:
            continue
        roc_sorted = sorted(roc, key=lambda r: r["fpr"])
        fprs = [r["fpr"] for r in roc_sorted]
        tprs = [r["tpr"] for r in roc_sorted]
        ax.plot(fprs, tprs, style, markersize=3, label=f"T={T}")
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"Empirical audit ROC, sigma={sigma_for_roc}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fast smoke test (N=200)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    N = 200 if args.quick else 2000

    t0 = time.time()
    rows, roc_records = run_grid(N=N, seed=args.seed)
    elapsed = time.time() - t0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_results(rows, RESULTS_DIR)

    plot_theory_vs_audit(rows, FIGURES_DIR / "theory_vs_audit.png")
    plot_ratio(rows, FIGURES_DIR / "audit_theory_ratio.png")
    plot_roc(roc_records, sigma_for_roc=sorted({r["sigma"] for r in rows})[1], out_path=FIGURES_DIR / "roc_curves.png")

    print(f"Ran {len(rows)} configs (N={N} trials/world each) in {elapsed:.1f}s")
    print(f"Results written to {RESULTS_DIR}")
    print(f"Figures written to {FIGURES_DIR}")
    for r in rows:
        print(
            f"sigma={r['sigma']:.1f} T={r['T']:>2} "
            f"epsilon_theory={r['epsilon_theory']:.4f} "
            f"eps_lower={r['eps_lower']:.4f} "
            f"ratio={r['ratio_audit_over_theory']:.3f}"
        )


if __name__ == "__main__":
    main()
