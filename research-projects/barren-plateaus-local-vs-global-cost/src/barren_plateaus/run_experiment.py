"""End-to-end driver: run the sweep, fit models, write results.csv /
summary.json, and generate all plots into ../results.

Usage: python -m barren_plateaus.run_experiment
"""

import csv
import json
import os
import time

from . import experiment, plots, theory

COST_TYPES = ["global", "local"]
DEPTHS = [1, 2, 4, 8, 16]
N_VALUES = [2, 4, 6, 8, 10, 12, 14]
NUM_SAMPLES = 250
BASE_SEED = 20260707

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    print(f"Running sweep: {len(COST_TYPES)} cost types x {len(DEPTHS)} depths x {len(N_VALUES)} widths "
          f"x {NUM_SAMPLES} samples...")
    results = experiment.run_sweep(COST_TYPES, DEPTHS, N_VALUES, NUM_SAMPLES, BASE_SEED)
    print(f"Sweep done in {time.time() - t0:.1f}s ({len(results)} configurations).")

    # results.csv
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cost_type", "depth", "n", "num_samples", "mean_gradient", "variance", "variance_stderr"])
        for r in results:
            writer.writerow([r.cost_type, r.depth, r.n, r.num_samples, r.mean, r.variance, r.variance_stderr])
    print(f"Wrote {csv_path}")

    # per-(cost_type, depth) exponential + power-law fits
    fit_table = []
    for cost_type in COST_TYPES:
        for depth in DEPTHS:
            rows = sorted([r for r in results if r.cost_type == cost_type and r.depth == depth], key=lambda r: r.n)
            ns = [r.n for r in rows]
            var = [r.variance for r in rows]
            exp_fit = theory.fit_exponential(ns, var)
            pow_fit = theory.fit_power_law(ns, var)
            fit_table.append(
                {
                    "cost_type": cost_type,
                    "depth": depth,
                    "exp_slope": exp_fit.slope,
                    "exp_intercept": exp_fit.intercept,
                    "exp_r2": exp_fit.r2,
                    "pow_slope": pow_fit.slope,
                    "pow_intercept": pow_fit.intercept,
                    "pow_r2": pow_fit.r2,
                    "preferred_model": "exponential" if exp_fit.r2 >= pow_fit.r2 else "power_law",
                }
            )

    summary = {
        "config": {
            "cost_types": COST_TYPES,
            "depths": DEPTHS,
            "n_values": N_VALUES,
            "num_samples": NUM_SAMPLES,
            "base_seed": BASE_SEED,
        },
        "fits": fit_table,
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # Plots
    plots.plot_variance_vs_n(results, DEPTHS, os.path.join(RESULTS_DIR, "variance_vs_n.png"))
    plots.plot_fit_slopes(fit_table, os.path.join(RESULTS_DIR, "fit_slopes_vs_depth.png"))
    plots.plot_model_comparison(results, depth=1, out_path=os.path.join(RESULTS_DIR, "model_comparison_depth1.png"))
    plots.plot_model_comparison(results, depth=16, out_path=os.path.join(RESULTS_DIR, "model_comparison_depth16.png"))
    print(f"Wrote plots to {RESULTS_DIR}")

    print(f"Total wall time: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
