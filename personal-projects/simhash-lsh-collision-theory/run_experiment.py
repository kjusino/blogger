#!/usr/bin/env python3
"""Entry point: run all three experiments, save results/*.csv,
results/summary.json, and figures/*.png."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from src import theory
from src.experiment import (
    find_empirical_threshold_angle,
    fit_power_law_exponent,
    run_and_or_threshold_experiment,
    run_scaling_experiment,
    run_single_hash_experiment,
)
from src.plotting import plot_and_or_threshold, plot_scaling, plot_single_hash_collision

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

SEED = 20260709


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary: dict = {}

    # ---- Experiment 1: single-hash collision probability ----
    print("Running experiment 1: single-hash collision probability...")
    thetas_1 = list(np.linspace(0.15, math.pi - 0.15, 12))
    dims_1 = [8, 64, 512]
    df1 = run_single_hash_experiment(thetas_1, dims_1, num_trials=20000, seed=SEED)
    df1.to_csv(os.path.join(RESULTS_DIR, "exp1_single_hash_collision.csv"), index=False)
    plot_single_hash_collision(df1, os.path.join(FIGURES_DIR, "exp1_single_hash_collision.png"))
    max_z = float(df1["z_score"].max())
    max_abs_error = float(df1["abs_error"].max())
    summary["experiment_1_single_hash"] = {
        "description": "Empirical Pr[random hyperplane agrees] vs theory 1 - theta/pi",
        "n_points": len(df1),
        "dims_tested": dims_1,
        "num_trials_per_point": 20000,
        "max_abs_error": max_abs_error,
        "max_z_score": max_z,
        "verdict": "PASS" if max_z < 5.0 else "FAIL",
        "verdict_rule": "max |empirical - theory| / stderr < 5.0 across all (theta, dim) points",
    }
    print(f"  max_abs_error={max_abs_error:.5f}, max_z_score={max_z:.2f}")

    # ---- Experiment 2: AND-OR S-curve threshold ----
    print("Running experiment 2: AND-OR LSH S-curve threshold...")
    k2, L2 = 8, 20
    theory_threshold = theory.threshold_angle(k2, L2)
    thetas_2 = list(np.linspace(0.3, 2.0, 18))
    df2 = run_and_or_threshold_experiment(
        k=k2, L=L2, thetas=thetas_2, num_trials=2000, dim=64, seed=SEED
    )
    df2.to_csv(os.path.join(RESULTS_DIR, "exp2_and_or_threshold.csv"), index=False)
    empirical_threshold = find_empirical_threshold_angle(df2)
    plot_and_or_threshold(
        df2, k2, L2, empirical_threshold, theory_threshold,
        os.path.join(FIGURES_DIR, "exp2_and_or_threshold.png"),
    )
    threshold_error = abs(empirical_threshold - theory_threshold)
    max_curve_abs_error = float(df2["abs_error"].max())
    summary["experiment_2_and_or_threshold"] = {
        "description": "Empirical AND-OR (k, L) LSH recall S-curve vs theory 1-(1-p^k)^L",
        "k": k2,
        "L": L2,
        "num_trials_per_point": 2000,
        "theory_threshold_theta": theory_threshold,
        "empirical_threshold_theta": empirical_threshold,
        "threshold_abs_error_radians": threshold_error,
        "max_curve_abs_error": max_curve_abs_error,
        "verdict": "PASS" if threshold_error < 0.08 and max_curve_abs_error < 0.08 else "FAIL",
        "verdict_rule": "threshold error < 0.08 rad AND max curve abs error < 0.08",
    }
    print(f"  theory_threshold={theory_threshold:.4f}, empirical_threshold={empirical_threshold:.4f}, "
          f"error={threshold_error:.4f}")

    # ---- Experiment 3: sublinear query-cost scaling exponent ----
    print("Running experiment 3: sublinear query-cost scaling (Indyk-Motwani rho)...")
    near_theta, far_theta = math.pi / 6, math.pi / 2
    p1 = theory.single_hash_collision_prob(near_theta)
    p2 = theory.single_hash_collision_prob(far_theta)
    rho_theory = theory.rho_exponent(p1, p2)
    n_list = [200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
    trials_per_n = 60
    df3 = run_scaling_experiment(
        near_theta=near_theta,
        far_theta=far_theta,
        dim=32,
        n_list=n_list,
        trials_per_n=trials_per_n,
        seed=SEED,
        k=None,  # k, L computed per-n inside the function via k_of_n/L_of_n
        L=None,
    )
    df3.to_csv(os.path.join(RESULTS_DIR, "exp3_scaling.csv"), index=False)
    rho_hat, r_squared = fit_power_law_exponent(df3["n"].to_numpy(), df3["mean_candidates"].to_numpy())
    plot_scaling(df3, rho_hat, rho_theory, os.path.join(FIGURES_DIR, "exp3_scaling.png"))
    rel_error = abs(rho_hat - rho_theory) / rho_theory
    summary["experiment_3_scaling"] = {
        "description": "Fitted exponent of mean LSH candidate-set size vs n, compared to Indyk-Motwani rho = ln(1/p1)/ln(1/p2)",
        "near_theta": near_theta,
        "far_theta": far_theta,
        "p1_single_hash": p1,
        "p2_single_hash": p2,
        "rho_theory": rho_theory,
        "rho_fitted": rho_hat,
        "fit_r_squared": r_squared,
        "relative_error": rel_error,
        "n_list": n_list,
        "trials_per_n": trials_per_n,
        "verdict": "PASS" if rel_error < 0.25 and r_squared > 0.8 else "FAIL",
        "verdict_rule": "|rho_fitted - rho_theory| / rho_theory < 0.25 AND R^2 > 0.8",
    }
    print(f"  rho_theory={rho_theory:.4f}, rho_fitted={rho_hat:.4f}, R^2={r_squared:.4f}")

    overall_pass = all(v["verdict"] == "PASS" for v in summary.values())
    summary["overall_verdict"] = "PASS" if overall_pass else "PARTIAL/FAIL"
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOverall verdict: {summary['overall_verdict']}")
    print(f"Results written to {RESULTS_DIR}, figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
