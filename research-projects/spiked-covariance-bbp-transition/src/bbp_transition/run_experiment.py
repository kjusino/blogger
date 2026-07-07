"""End-to-end driver: run all sweeps, write results/, generate all plots.

Usage:
    python -m bbp_transition.run_experiment
"""

from __future__ import annotations

import csv
import json
import os

from .experiment import sweep_lambda_grid, estimate_detection_threshold, finite_size_scaling
from .theory import bbp_threshold
from . import plots as plotmod

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

MAIN_P = 150
MAIN_C_VALUES = [0.1, 0.3, 0.5, 0.7]
MAIN_LAM_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
MAIN_TRIALS = 50
MAIN_SEED = 1

CROSSING_P = 150
CROSSING_C_VALUES = [0.2, 0.5, 0.8]
CROSSING_LAM_RATIOS = [0.1 + 0.1 * i for i in range(25)]  # 0.1 .. 2.5 in steps of 0.1
CROSSING_TRIALS = 40
CROSSING_SEED = 2

SCALING_C = 0.3
SCALING_P_VALUES = [50, 100, 200, 400, 800]
SCALING_TRIALS = 80
SCALING_SEED = 3


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running main (c, lam) grid sweep...")
    main_results = sweep_lambda_grid(
        p=MAIN_P,
        c_values=MAIN_C_VALUES,
        lam_ratios=MAIN_LAM_RATIOS,
        trials=MAIN_TRIALS,
        seed=MAIN_SEED,
    )

    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(main_results[0].to_dict().keys()))
        writer.writeheader()
        for r in main_results:
            writer.writerow(r.to_dict())
    print(f"Wrote {results_csv} ({len(main_results)} rows)")

    print("Estimating empirical detection thresholds...")
    crossing_results = []
    for c in CROSSING_C_VALUES:
        lam_hat, thr, lams, aligns = estimate_detection_threshold(
            p=CROSSING_P,
            c=c,
            lam_ratios_fine=CROSSING_LAM_RATIOS,
            trials=CROSSING_TRIALS,
            seed=CROSSING_SEED,
        )
        crossing_results.append((c, lam_hat, thr, lams, aligns))
        print(f"  c={c}: theory lam*={thr:.4f}, empirical lam_hat={lam_hat}")

    print("Running finite-size scaling sweep...")
    below_rows = finite_size_scaling(
        c=SCALING_C, lam_ratio=0.5, p_values=SCALING_P_VALUES, trials=SCALING_TRIALS, seed=SCALING_SEED
    )
    above_rows = finite_size_scaling(
        c=SCALING_C, lam_ratio=2.0, p_values=SCALING_P_VALUES, trials=SCALING_TRIALS, seed=SCALING_SEED + 1
    )

    print("Generating plots...")
    plotmod.plot_eigenvalue_vs_lambda(main_results, c_target=0.3, out_path=os.path.join(RESULTS_DIR, "eigenvalue_vs_lambda_c0.3.png"))
    plotmod.plot_alignment_vs_lambda(main_results, c_target=0.3, out_path=os.path.join(RESULTS_DIR, "alignment_vs_lambda_c0.3.png"))
    plotmod.plot_phase_diagram(main_results, out_path=os.path.join(RESULTS_DIR, "phase_diagram.png"))
    plotmod.plot_threshold_crossing(crossing_results, out_path=os.path.join(RESULTS_DIR, "threshold_crossing.png"))
    plotmod.plot_finite_size_scaling(
        {"below threshold (lam=0.5*lam*)": below_rows, "above threshold (lam=2*lam*)": above_rows},
        out_path=os.path.join(RESULTS_DIR, "finite_size_scaling.png"),
    )

    max_rel_err_eig = max(r.rel_err_eig for r in main_results)
    max_abs_err_align = max(r.abs_err_align for r in main_results)
    threshold_errors = [
        abs(lam_hat - thr) / thr for (c, lam_hat, thr, _, _) in crossing_results if lam_hat is not None
    ]

    summary = {
        "main_sweep": {
            "p": MAIN_P,
            "c_values": MAIN_C_VALUES,
            "lam_ratios": MAIN_LAM_RATIOS,
            "trials_per_cell": MAIN_TRIALS,
            "n_cells": len(main_results),
            "max_rel_err_top_eigenvalue": max_rel_err_eig,
            "max_abs_err_alignment": max_abs_err_align,
        },
        "detection_threshold": [
            {
                "c": c,
                "theory_lam_star": thr,
                "empirical_lam_hat": lam_hat,
                "rel_err": (abs(lam_hat - thr) / thr) if lam_hat is not None else None,
            }
            for (c, lam_hat, thr, _, _) in crossing_results
        ],
        "max_threshold_rel_err": max(threshold_errors) if threshold_errors else None,
        "finite_size_scaling": {
            "c": SCALING_C,
            "p_values": SCALING_P_VALUES,
            "trials_per_cell": SCALING_TRIALS,
            "below_threshold_abs_err_by_p": {r["p"]: r["abs_err_eig"] for r in below_rows},
            "above_threshold_abs_err_by_p": {r["p"]: r["abs_err_eig"] for r in above_rows},
        },
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
