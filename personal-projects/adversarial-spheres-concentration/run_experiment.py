"""Driver: sweep the concentric-spheres dimension grid, train a classifier at each
d, attack it three ways, compare against the exact Levy-concentration ceiling,
fit power laws, save CSV results, and render figures.

Usage: python run_experiment.py
"""

import csv
import os

import numpy as np

from src.experiment import run_dimension, summarize_rows
from src.stats_utils import fit_power_law
from src.concentration import levy_ceiling_exact
from src import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

DIMENSION_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024]
R_INNER, R_OUTER = 1.0, 1.3


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    summary_rows = []
    raw_rows = []

    for d in DIMENSION_GRID:
        print(f"=== d={d} ===", flush=True)
        result = run_dimension(
            d,
            r_inner=R_INNER,
            r_outer=R_OUTER,
            n_train_per_class=5000,
            n_val_per_class=800,
            n_test_per_class=1200,
            n_attack_per_sphere=40,
            hidden=128,
            epochs=600,
            seed=0,
        )
        for row in result["rows"]:
            raw_rows.append(row)

        on_sphere = summarize_rows(result["rows"], "on_sphere_dist")
        general_l2 = summarize_rows(result["rows"], "general_l2_dist")
        radial = summarize_rows(result["rows"], "radial_dist")

        summary_rows.append({
            "d": d,
            "test_acc": result["test_acc"],
            "p_minor_inner": result["p_minor_inner"],
            "p_minor_outer": result["p_minor_outer"],
            "ceiling_inner": result["ceiling_inner"],
            "ceiling_outer": result["ceiling_outer"],
            "on_sphere_median": on_sphere["median"],
            "on_sphere_p25": on_sphere["p25"],
            "on_sphere_p75": on_sphere["p75"],
            "on_sphere_found_frac": on_sphere["found_frac"],
            "general_l2_median": general_l2["median"],
            "general_l2_p25": general_l2["p25"],
            "general_l2_p75": general_l2["p75"],
            "general_l2_found_frac": general_l2["found_frac"],
            "radial_median": radial["median"],
            "radial_found_frac": radial["found_frac"],
        })
        print(f"    acc={result['test_acc']:.3f}  on_sphere_median={on_sphere['median']:.5f}  "
              f"ceiling(avg)={(result['ceiling_inner']+result['ceiling_outer'])/2:.5f}", flush=True)

    write_csv(
        os.path.join(RESULTS_DIR, "summary_by_dimension.csv"),
        summary_rows,
        fieldnames=list(summary_rows[0].keys()),
    )
    write_csv(
        os.path.join(RESULTS_DIR, "raw_attack_rows.csv"),
        raw_rows,
        fieldnames=list(raw_rows[0].keys()),
    )

    summary = {k: np.array([row[k] for row in summary_rows]) for k in summary_rows[0].keys()}

    # Isolate the pure isoperimetric d-dependence from the confound of p_minor
    # itself drifting with d (because training difficulty changes with d): fix
    # the minority measure at a representative constant and recompute the
    # closed-form ceiling across the same dimension grid. This has no empirical
    # noise at all -- it is a direct evaluation of the formula in concentration.py
    # -- so its fitted exponent is a validation of the theory's asymptotic rate,
    # separate from the "realized" ceiling that tracks each dimension's own
    # (noisier) empirically-trained classifier.
    fixed_p_minor = 0.3
    avg_radius = (R_INNER + R_OUTER) / 2
    fixed_p_ceiling = np.array([
        levy_ceiling_exact(fixed_p_minor, int(d), avg_radius) for d in summary["d"]
    ])

    fits = {
        "on_sphere": fit_power_law(summary["d"], summary["on_sphere_median"]),
        "general_l2": fit_power_law(summary["d"], summary["general_l2_median"]),
        "ceiling_realized": fit_power_law(summary["d"], (summary["ceiling_inner"] + summary["ceiling_outer"]) / 2),
        "ceiling_fixed_p_minor": fit_power_law(summary["d"], fixed_p_ceiling),
    }
    write_csv(
        os.path.join(RESULTS_DIR, "power_law_fits.csv"),
        [
            {
                "quantity": k,
                "exponent": v["exponent"],
                "exponent_ci95_lo": v["exponent_ci95"][0],
                "exponent_ci95_hi": v["exponent_ci95"][1],
                "c": v["c"],
                "r_squared": v["r_squared"],
                "n": v["n"],
            }
            for k, v in fits.items()
        ],
        fieldnames=["quantity", "exponent", "exponent_ci95_lo", "exponent_ci95_hi", "c", "r_squared", "n"],
    )

    for k, v in fits.items():
        print(f"fit[{k}]: exponent={v['exponent']:.3f} "
              f"95% CI=({v['exponent_ci95'][0]:.3f}, {v['exponent_ci95'][1]:.3f}) R^2={v['r_squared']:.3f}")

    plotting.plot_accuracy(summary, os.path.join(FIGURES_DIR, "accuracy_vs_dimension.png"))
    plotting.plot_robustness_vs_dimension(summary, fits, os.path.join(FIGURES_DIR, "robustness_vs_dimension.png"))
    plotting.plot_exponent_estimates(fits, os.path.join(FIGURES_DIR, "exponent_estimates.png"))
    plotting.plot_attack_type_comparison(summary, os.path.join(FIGURES_DIR, "attack_type_comparison.png"))
    plotting.plot_ceiling_decomposition(summary, fixed_p_ceiling, fixed_p_minor, fits,
                                         os.path.join(FIGURES_DIR, "ceiling_decomposition.png"))

    raw_dict = {k: np.array([row[k] for row in raw_rows]) for k in raw_rows[0].keys()}
    MIN_SAMPLES_FOR_HISTOGRAM = 20
    counts_per_d = {}
    for row in raw_rows:
        if row["on_sphere_found"]:
            counts_per_d[row["d"]] = counts_per_d.get(row["d"], 0) + 1
    dims_with_enough_data = sorted(d for d, n in counts_per_d.items() if n >= MIN_SAMPLES_FOR_HISTOGRAM)
    low_d = dims_with_enough_data[0] if dims_with_enough_data else DIMENSION_GRID[0]
    high_d = dims_with_enough_data[-1] if dims_with_enough_data else DIMENSION_GRID[-1]
    plotting.plot_distance_distributions(raw_dict, low_d, high_d,
                                          os.path.join(FIGURES_DIR, "distance_distributions.png"))

    print("Done. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
