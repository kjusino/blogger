#!/usr/bin/env python3
"""Run the full 2D Ising finite-size-scaling study and produce results + figures.

Usage:
    python3 run_experiment.py            # full grid (a few minutes)
    python3 run_experiment.py --quick    # small smoke-test grid (seconds)
"""
import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import theory
from src.experiment import (
    run_fss_grid,
    analyze_scaling,
    onsager_validation,
    autocorrelation_comparison,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_magnetization_and_susceptibility(by_L, L_values, Tc_exact, path_m, path_chi):
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in L_values:
        ax.errorbar(by_L[L]["T"], by_L[L]["m"], fmt="o-", label=f"L={L}", markersize=4)
    ax.axvline(Tc_exact, color="k", linestyle="--", alpha=0.6, label=r"exact $T_c$")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle |m| \rangle$")
    ax.set_title("Magnetization per spin vs. temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_m, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for L in L_values:
        ax.plot(by_L[L]["T"], by_L[L]["chi"], "o-", label=f"L={L}", markersize=4)
    ax.axvline(Tc_exact, color="k", linestyle="--", alpha=0.6, label=r"exact $T_c$")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\chi$ (susceptibility per spin)")
    ax.set_title("Susceptibility vs. temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_chi, dpi=150)
    plt.close(fig)


def plot_binder_cumulant(by_L, L_values, Tc_exact, tc_estimate, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in L_values:
        ax.plot(by_L[L]["T"], by_L[L]["u4"], "o-", label=f"L={L}", markersize=4)
    ax.axvline(Tc_exact, color="k", linestyle="--", alpha=0.6, label=r"exact $T_c$")
    if tc_estimate is not None:
        ax.axvline(tc_estimate, color="r", linestyle=":", alpha=0.8, label=r"crossing $T_c$ estimate")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"Binder cumulant $U_4$")
    ax.set_title("Binder cumulant crossing (Tc estimator)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scaling_collapse(scaling_result, L_values, path_m, path_chi):
    fig, ax = plt.subplots(figsize=(7, 5))
    for L in L_values:
        ax.plot(scaling_result["x_by_L"][L], scaling_result["y_by_L"][L], "o-", label=f"L={L}", markersize=4)
    ax.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
    ax.set_ylabel(r"$|m|\, L^{\beta/\nu}$")
    ax.set_title("Finite-size scaling collapse: magnetization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_m, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for L in L_values:
        ax.plot(scaling_result["x_by_L"][L], scaling_result["chi_y_by_L"][L], "o-", label=f"L={L}", markersize=4)
    ax.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
    ax.set_ylabel(r"$\chi\, L^{-\gamma/\nu}$")
    ax.set_title("Finite-size scaling collapse: susceptibility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_chi, dpi=150)
    plt.close(fig)


def plot_onsager_validation(rows, path):
    Ts = [r["T"] for r in rows]
    e_sim = [r["energy_sim"] for r in rows]
    e_exact = [r["energy_exact"] for r in rows]
    m_sim = [r["magnetization_sim"] for r in rows]
    m_exact = [r["magnetization_exact"] for r in rows]

    fine_T = np.linspace(min(Ts) - 0.05, max(Ts) + 0.05, 400)
    fine_T = fine_T[fine_T > 0.05]
    fine_e = theory.onsager_energy(fine_T)
    fine_m = theory.onsager_magnetization(fine_T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fine_T, fine_e, "-", color="gray", label="exact (Onsager)")
    axes[0].plot(Ts, e_sim, "o", color="C0", label="Metropolis (simulated)")
    axes[0].plot(Ts, e_exact, "x", color="C1", label="exact (sampled points)")
    axes[0].set_xlabel("Temperature T")
    axes[0].set_ylabel("Energy per spin")
    axes[0].set_title("Energy: simulation vs. exact Onsager solution")
    axes[0].legend()

    axes[1].plot(fine_T, fine_m, "-", color="gray", label="exact (Onsager-Yang)")
    axes[1].plot(Ts, m_sim, "o", color="C0", label="Metropolis (simulated)")
    axes[1].plot(Ts, m_exact, "x", color="C1", label="exact (sampled points)")
    axes[1].set_xlabel("Temperature T")
    axes[1].set_ylabel("Magnetization per spin")
    axes[1].set_title("Magnetization: simulation vs. exact Onsager-Yang solution")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_autocorrelation_comparison(rows, path):
    Ts = [r["T"] for r in rows]
    met_tau = [r["metropolis_tau_sweeps"] for r in rows]
    wolf_tau = [r["wolff_tau_steps"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ts, met_tau, "o-", label="Metropolis (per sweep)")
    ax.plot(Ts, wolf_tau, "s-", label="Wolff (per cluster step)")
    ax.axvline(theory.T_C, color="k", linestyle="--", alpha=0.6, label=r"exact $T_c$")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"Integrated autocorrelation time $\tau_{int}$")
    ax.set_yscale("log")
    ax.set_title("Critical slowing down: Metropolis vs. Wolff")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a fast smoke-test grid")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t_start = time.time()

    if args.quick:
        L_values = [8, 16]
        T_grid = np.linspace(2.0, 2.5, 6)
        n_equil, n_sample, sample_interval, n_seeds = 100, 100, 1, 2
        onsager_T_values = [1.0, 2.0, 3.0]
        onsager_L = 16
        onsager_n_equil, onsager_n_sample, onsager_n_seeds = 100, 100, 2
        autocorr_T_values = [theory.T_C - 0.2, theory.T_C, theory.T_C + 0.2]
        autocorr_L = 16
        autocorr_n_equil, autocorr_n_sample = 50, 200
    else:
        L_values = [8, 16, 32, 64]
        T_grid = np.concatenate(
            [
                np.linspace(1.6, 2.1, 6),
                np.linspace(2.1, 2.45, 14),
                np.linspace(2.45, 3.0, 6),
            ]
        )
        T_grid = np.unique(np.round(T_grid, 6))
        n_equil, n_sample, sample_interval, n_seeds = 2000, 2000, 2, 4
        onsager_T_values = [0.8, 1.2, 1.6, 2.0, 2.269185314213022, 2.5, 3.0, 4.0]
        onsager_L = 64
        onsager_n_equil, onsager_n_sample, onsager_n_seeds = 2000, 2000, 4
        autocorr_T_values = list(np.linspace(theory.T_C - 0.4, theory.T_C + 0.4, 7))
        autocorr_L = 32
        autocorr_n_equil, autocorr_n_sample = 500, 3000

    print(f"[1/4] Finite-size scaling grid: L={L_values}, {len(T_grid)} T points, {n_seeds} seeds")
    rows, by_L = run_fss_grid(L_values, T_grid, n_equil, n_sample, sample_interval, n_seeds)
    write_csv(
        rows,
        os.path.join(RESULTS_DIR, "fss_grid.csv"),
        fieldnames=[
            "L", "T", "m_mean", "m_std", "chi_mean", "chi_std",
            "c_mean", "c_std", "u4_mean", "u4_std", "e_mean", "e_std",
        ],
    )

    print("[2/4] Scaling analysis (Binder crossing + data collapse)")
    scaling_result = analyze_scaling(by_L, L_values)

    print(f"[3/4] Onsager exact-solution validation at L={onsager_L}")
    onsager_rows = onsager_validation(
        onsager_L, onsager_T_values, onsager_n_equil, onsager_n_sample, 1, onsager_n_seeds
    )
    write_csv(
        onsager_rows,
        os.path.join(RESULTS_DIR, "onsager_validation.csv"),
        fieldnames=[
            "T", "L", "energy_sim", "energy_exact", "energy_abs_error",
            "magnetization_sim", "magnetization_exact", "magnetization_abs_error",
        ],
    )

    print(f"[4/4] Metropolis vs. Wolff autocorrelation comparison at L={autocorr_L}")
    autocorr_rows = autocorrelation_comparison(
        autocorr_L, autocorr_T_values, autocorr_n_equil, autocorr_n_sample
    )
    write_csv(
        autocorr_rows,
        os.path.join(RESULTS_DIR, "autocorrelation_comparison.csv"),
        fieldnames=[
            "T", "L", "metropolis_tau_sweeps", "wolff_tau_steps",
            "metropolis_wall_seconds", "wolff_wall_seconds",
            "metropolis_seconds_per_independent_sample", "wolff_seconds_per_independent_sample",
            "mean_wolff_cluster_size",
        ],
    )

    print("Generating figures...")
    plot_magnetization_and_susceptibility(
        by_L, L_values, theory.T_C,
        os.path.join(FIGURES_DIR, "magnetization_vs_T.png"),
        os.path.join(FIGURES_DIR, "susceptibility_vs_T.png"),
    )
    plot_binder_cumulant(
        by_L, L_values, theory.T_C, scaling_result["tc_estimate"]["Tc_estimate"],
        os.path.join(FIGURES_DIR, "binder_cumulant_crossing.png"),
    )
    plot_scaling_collapse(
        scaling_result, L_values,
        os.path.join(FIGURES_DIR, "scaling_collapse_magnetization.png"),
        os.path.join(FIGURES_DIR, "scaling_collapse_susceptibility.png"),
    )
    plot_onsager_validation(onsager_rows, os.path.join(FIGURES_DIR, "onsager_validation.png"))
    plot_autocorrelation_comparison(
        autocorr_rows, os.path.join(FIGURES_DIR, "autocorrelation_comparison.png")
    )

    elapsed = time.time() - t_start

    max_onsager_energy_error = max(r["energy_abs_error"] for r in onsager_rows)
    max_wolff_speedup = max(
        r["metropolis_seconds_per_independent_sample"] / r["wolff_seconds_per_independent_sample"]
        for r in autocorr_rows
    )

    summary = {
        "elapsed_seconds": elapsed,
        "quick_mode": args.quick,
        "L_values": L_values,
        "n_T_points": len(T_grid),
        "n_seeds": n_seeds,
        "exact_Tc": theory.T_C,
        "estimated_Tc": scaling_result["tc_estimate"]["Tc_estimate"],
        "Tc_relative_error": scaling_result["tc_relative_error"],
        "individual_crossings": scaling_result["tc_estimate"]["crossings"],
        "raw_collapse_rmse": scaling_result["raw_collapse_rmse"],
        "rescaled_magnetization_collapse_rmse": scaling_result["rescaled_magnetization_collapse_rmse"],
        "rescaled_susceptibility_collapse_rmse": scaling_result["rescaled_susceptibility_collapse_rmse"],
        "collapse_improvement_factor": scaling_result["collapse_improvement_factor"],
        "max_onsager_energy_abs_error": max_onsager_energy_error,
        "max_wolff_vs_metropolis_speedup": max_wolff_speedup,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nDone in {elapsed:.1f}s. See results/ and figures/.")


if __name__ == "__main__":
    main()
