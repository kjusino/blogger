"""Orchestrates the three studies in this project:

  1. Theory-vs-brute-force validation grid over (b, lambda).
  2. The classical-2-competitive convergence sanity check across a wide
     range of b (lambda = 0).
  3. Monte Carlo expected-ratio sweep over (sigma, lambda), the empirical
     argmin lambda*(sigma), and a comparison against a simple closed-form
     heuristic lambda*_approx(sigma).

Writes CSVs and a summary.json under results/, and PNG figures under
figures/. Designed to be called from run_experiment.py, either in "full"
mode (the real experiment grid) or "--quick" mode (a fast smoke test).
"""

from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .algorithm import tau
from .theory import (
    robustness_formula,
    consistency_formula,
    robustness_tolerance,
    consistency_tolerance,
)
from .brute_force import brute_force_robustness, brute_force_consistency
from .monte_carlo import expected_ratio, argmin_lambda, lambda_star_approx, fit_c
from .predictor import sample_x_lognormal


# --------------------------------------------------------------------------
# 1. Theory vs. brute force
# --------------------------------------------------------------------------


def run_theory_vs_bruteforce(b_grid, lambda_grid):
    """For every (b, lam) in the grid, compute Robustness and Consistency
    both via the closed form and via brute-force search, and record the
    discrepancy against the tolerance derived in theory.py."""
    rows = []
    for b in b_grid:
        for lam in lambda_grid:
            r_theory = robustness_formula(lam, b)
            r_bf, r_bf_x, r_bf_y = brute_force_robustness(lam, b)
            r_diff = abs(r_theory - r_bf)
            r_tol = robustness_tolerance(lam, b)

            c_theory = consistency_formula(lam, b)
            c_bf, c_bf_x = brute_force_consistency(lam, b)
            c_diff = abs(c_theory - c_bf)
            c_tol = consistency_tolerance(lam, b)

            rows.append(
                {
                    "b": b,
                    "lam": lam,
                    "robustness_theory": r_theory,
                    "robustness_bruteforce": r_bf,
                    "robustness_bf_argmax_x": r_bf_x,
                    "robustness_bf_argmax_y": r_bf_y,
                    "robustness_abs_diff": r_diff,
                    "robustness_tolerance": r_tol,
                    "robustness_within_tolerance": r_diff <= r_tol,
                    "consistency_theory": c_theory,
                    "consistency_bruteforce": c_bf,
                    "consistency_bf_argmax_x": c_bf_x,
                    "consistency_abs_diff": c_diff,
                    "consistency_tolerance": c_tol,
                    "consistency_within_tolerance": c_diff <= c_tol,
                }
            )
    return rows


def write_theory_vs_bruteforce_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# 2. Classical 2-competitive convergence (lambda = 0, wide range of b)
# --------------------------------------------------------------------------


def run_robustness_vs_b(b_grid_wide, bf_grid_rows=None):
    """Robustness at lambda = 0 (the classical no-predictions strategy)
    across a wide range of b, via the closed form. bf_grid_rows (if given,
    from run_theory_vs_bruteforce with lam=0 present) is used to attach
    brute-force cross-checks at whichever b values overlap."""
    bf_lookup = {}
    if bf_grid_rows:
        for row in bf_grid_rows:
            if row["lam"] == 0.0:
                bf_lookup[row["b"]] = row["robustness_bruteforce"]

    rows = []
    for b in b_grid_wide:
        theory = robustness_formula(0.0, b)
        rows.append(
            {
                "b": b,
                "robustness_theory_lambda0": theory,
                "robustness_bruteforce_lambda0": bf_lookup.get(b, ""),
            }
        )
    return rows


def write_robustness_vs_b_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# 3. Monte Carlo sigma/lambda sweep + lambda* comparison
# --------------------------------------------------------------------------


def run_monte_carlo_sweep(b, sigma_grid, lambda_grid, n_samples, rng, x_sampler=sample_x_lognormal):
    """For every (sigma, lam), estimate the expected competitive ratio via
    Monte Carlo. Returns a flat list of rows plus a dict sigma -> (best_lambda, ratios array)."""
    rows = []
    best_by_sigma = {}
    for sigma in sigma_grid:
        best_lam, ratios = argmin_lambda(b, sigma, lambda_grid, n_samples, rng, x_sampler)
        best_by_sigma[sigma] = (best_lam, ratios)
        for lam, ratio in zip(lambda_grid, ratios):
            rows.append({"sigma": sigma, "lam": float(lam), "expected_ratio": float(ratio)})
    return rows, best_by_sigma


def write_monte_carlo_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_lambda_star_comparison(sigma_grid, best_by_sigma):
    """Fit the simple heuristic lambda*_approx(sigma) = exp(-c*sigma) from
    the largest-sigma empirical anchor point, then compare it to the
    empirical argmin at every sigma in the grid."""
    sigma_grid_sorted = sorted(sigma_grid)
    sigma_anchor = sigma_grid_sorted[-1]
    lambda_star_anchor = best_by_sigma[sigma_anchor][0]

    if sigma_anchor > 0:
        c = fit_c(sigma_anchor, lambda_star_anchor)
    else:
        c = 0.0

    rows = []
    for sigma in sigma_grid_sorted:
        empirical = best_by_sigma[sigma][0]
        approx = lambda_star_approx(sigma, c)
        abs_err = abs(empirical - approx)
        rel_err = abs_err / empirical if empirical > 0 else float("nan")
        rows.append(
            {
                "sigma": sigma,
                "lambda_star_empirical": empirical,
                "lambda_star_approx": approx,
                "abs_error": abs_err,
                "rel_error": rel_err,
            }
        )
    return rows, c


def write_lambda_star_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_tradeoff_figure(b_representative, lambda_grid, bf_rows, path):
    """Figure 1: Robustness(lam) and Consistency(lam) theory curves,
    overlaid with brute-force points, for a representative b."""
    lam_dense = np.linspace(0.0, 1.0, 201)
    rob_dense = [robustness_formula(lam, b_representative) for lam in lam_dense]
    con_dense = [consistency_formula(lam, b_representative) for lam in lam_dense]

    rows_b = [r for r in bf_rows if r["b"] == b_representative]
    rows_b.sort(key=lambda r: r["lam"])
    lam_pts = [r["lam"] for r in rows_b]
    rob_pts = [r["robustness_bruteforce"] for r in rows_b]
    con_pts = [r["consistency_bruteforce"] for r in rows_b]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lam_dense, rob_dense, label="Robustness (theory)", color="#1f77b4")
    ax.plot(lam_dense, con_dense, label="Consistency (theory)", color="#d62728")
    ax.scatter(lam_pts, rob_pts, marker="o", color="#1f77b4", label="Robustness (brute force)", zorder=5)
    ax.scatter(lam_pts, con_pts, marker="x", color="#d62728", label="Consistency (brute force)", zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("lambda (trust in prediction)")
    ax.set_ylabel("worst-case competitive ratio (log scale)")
    ax.set_title(f"Robustness-Consistency tradeoff (b = {b_representative})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_robustness_vs_b_figure(rows, path):
    """Figure 2: Robustness at lambda=0 vs. b, converging to 2."""
    b_vals = [r["b"] for r in rows]
    theory_vals = [r["robustness_theory_lambda0"] for r in rows]
    bf_b = [r["b"] for r in rows if r["robustness_bruteforce_lambda0"] != ""]
    bf_vals = [r["robustness_bruteforce_lambda0"] for r in rows if r["robustness_bruteforce_lambda0"] != ""]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(b_vals, theory_vals, marker="o", markersize=3, color="#1f77b4", label="Robustness(lambda=0, b) (theory)")
    if bf_b:
        ax.scatter(bf_b, bf_vals, marker="x", s=80, color="#d62728", label="brute force cross-check", zorder=5)
    ax.axhline(2.0, color="gray", linestyle="--", label="classical bound (ratio = 2)")
    ax.set_xscale("log")
    ax.set_xlabel("b (buy cost, log scale)")
    ax.set_ylabel("Robustness at lambda = 0")
    ax.set_title("lambda=0 recovers the classical 2-competitive bound as b grows", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_expected_ratio_figure(sigma_grid, lambda_grid, mc_rows, path):
    """Figure 3: expected ratio vs lambda, one curve per sigma."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    sigma_grid_sorted = sorted(sigma_grid)
    for i, sigma in enumerate(sigma_grid_sorted):
        rows_s = [r for r in mc_rows if r["sigma"] == sigma]
        rows_s.sort(key=lambda r: r["lam"])
        lam_vals = [r["lam"] for r in rows_s]
        ratio_vals = [r["expected_ratio"] for r in rows_s]
        color = cmap(i / max(1, len(sigma_grid_sorted) - 1))
        ax.plot(lam_vals, ratio_vals, marker=".", color=color, label=f"sigma = {sigma}")
    ax.set_xlabel("lambda (trust in prediction)")
    ax.set_ylabel("Monte Carlo expected competitive ratio")
    ax.set_title("Expected ratio vs. lambda, by predictor noise sigma")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_lambda_star_figure(lambda_star_rows, c, path):
    """Figure 4: empirical argmin lambda*(sigma) vs the closed-form heuristic."""
    rows_sorted = sorted(lambda_star_rows, key=lambda r: r["sigma"])
    sigma_vals = [r["sigma"] for r in rows_sorted]
    empirical_vals = [r["lambda_star_empirical"] for r in rows_sorted]

    sigma_dense = np.linspace(0.0, max(sigma_vals), 200)
    approx_dense = [lambda_star_approx(s, c) for s in sigma_dense]

    sign = "-" if c >= 0 else "+"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigma_dense, approx_dense, color="#2ca02c", label=f"lambda*_approx(sigma) = exp({sign}{abs(c):.3f} * sigma)")
    ax.scatter(sigma_vals, empirical_vals, color="#d62728", zorder=5, label="Monte Carlo argmin lambda*(sigma)")
    ax.set_xlabel("sigma (predictor noise)")
    ax.set_ylabel("lambda*")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Empirically-optimal lambda vs. the simple closed-form heuristic")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------


def run_full_experiment(results_dir, figures_dir, quick: bool = False, seed: int = 42):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    t0 = time.time()
    rng = np.random.default_rng(seed)

    if quick:
        b_grid = [10, 50]
        lambda_grid = [0.0, 0.5, 1.0]
        b_grid_wide = [2, 5, 10, 50, 200, 1000]
        mc_b = 50
        sigma_grid = [0.0, 0.5, 2.0]
        lambda_grid_mc = list(np.round(np.linspace(0.0, 1.0, 6), 3))
        n_samples = 500
    else:
        b_grid = [10, 50, 200, 1000]
        lambda_grid = list(np.round(np.linspace(0.0, 1.0, 11), 3))
        b_grid_wide = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        mc_b = 200
        sigma_grid = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
        lambda_grid_mc = list(np.round(np.linspace(0.0, 1.0, 21), 3))
        n_samples = 20000

    # --- Sanity check: classical tight 2-competitive bound at lambda=0 ---
    classical_check_b = 1000 if not quick else 50
    classical_robustness = robustness_formula(0.0, classical_check_b)

    # --- 1. Theory vs brute force ---
    print(f"[1/3] Theory-vs-brute-force grid over b={b_grid}, lambda={lambda_grid} ...")
    bf_rows = run_theory_vs_bruteforce(b_grid, lambda_grid)
    write_theory_vs_bruteforce_csv(bf_rows, os.path.join(results_dir, "theory_vs_bruteforce.csv"))

    max_r_diff_row = max(bf_rows, key=lambda r: r["robustness_abs_diff"])
    max_c_diff_row = max(bf_rows, key=lambda r: r["consistency_abs_diff"])
    all_within_tol = all(r["robustness_within_tolerance"] and r["consistency_within_tolerance"] for r in bf_rows)

    # --- 2. Robustness vs b (classical bound convergence) ---
    print(f"[1/3] Robustness-vs-b convergence over b={b_grid_wide} ...")
    rvb_rows = run_robustness_vs_b(b_grid_wide, bf_grid_rows=bf_rows)
    write_robustness_vs_b_csv(rvb_rows, os.path.join(results_dir, "robustness_vs_b.csv"))

    # --- 3. Monte Carlo sweep ---
    print(f"[2/3] Monte Carlo sweep: b={mc_b}, sigma={sigma_grid}, n_samples={n_samples} ...")
    mc_rows, best_by_sigma = run_monte_carlo_sweep(mc_b, sigma_grid, lambda_grid_mc, n_samples, rng)
    write_monte_carlo_csv(mc_rows, os.path.join(results_dir, "monte_carlo.csv"))

    print("[3/3] Fitting and comparing lambda*(sigma) heuristic ...")
    lambda_star_rows, c_fit = run_lambda_star_comparison(sigma_grid, best_by_sigma)
    write_lambda_star_csv(lambda_star_rows, os.path.join(results_dir, "lambda_star.csv"))

    # --- Figures ---
    print("Generating figures ...")
    b_representative = 200 if 200 in b_grid else b_grid[-1]
    make_tradeoff_figure(b_representative, lambda_grid, bf_rows, os.path.join(figures_dir, "robustness_consistency_tradeoff.png"))
    make_robustness_vs_b_figure(rvb_rows, os.path.join(figures_dir, "robustness_vs_b.png"))
    make_expected_ratio_figure(sigma_grid, lambda_grid_mc, mc_rows, os.path.join(figures_dir, "expected_ratio_vs_lambda_by_sigma.png"))
    make_lambda_star_figure(lambda_star_rows, c_fit, os.path.join(figures_dir, "lambda_star_vs_sigma.png"))

    elapsed = time.time() - t0

    # --- Summary ---
    lambda_star_monotonic = all(
        lambda_star_rows[i]["lambda_star_empirical"] >= lambda_star_rows[i + 1]["lambda_star_empirical"] - 1e-9
        for i in range(len(lambda_star_rows) - 1)
    )

    summary = {
        "quick_mode": quick,
        "elapsed_seconds": elapsed,
        "b_grid": b_grid,
        "lambda_grid": lambda_grid,
        "classical_bound_check": {
            "b": classical_check_b,
            "robustness_at_lambda0": classical_robustness,
            "note": "should be close to 2.0 (the classical tight 2-competitive bound)",
        },
        "theory_vs_bruteforce": {
            "all_within_tolerance": all_within_tol,
            "max_robustness_abs_diff": {
                "value": max_r_diff_row["robustness_abs_diff"],
                "at_b": max_r_diff_row["b"],
                "at_lambda": max_r_diff_row["lam"],
                "tolerance_used": max_r_diff_row["robustness_tolerance"],
            },
            "max_consistency_abs_diff": {
                "value": max_c_diff_row["consistency_abs_diff"],
                "at_b": max_c_diff_row["b"],
                "at_lambda": max_c_diff_row["lam"],
                "tolerance_used": max_c_diff_row["consistency_tolerance"],
            },
        },
        "monte_carlo": {
            "b": mc_b,
            "sigma_grid": sigma_grid,
            "n_samples": n_samples,
            "lambda_star_empirical_by_sigma": {
                str(s): best_by_sigma[s][0] for s in sigma_grid
            },
            "lambda_star_heuristic_c": c_fit,
            "lambda_star_monotonic_nonincreasing": lambda_star_monotonic,
            "lambda_star_comparison": lambda_star_rows,
        },
    }

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
