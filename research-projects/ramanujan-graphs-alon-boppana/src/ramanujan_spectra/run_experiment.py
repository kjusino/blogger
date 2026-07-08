"""End-to-end driver: exact-spectrum validation, the full random-graph
sweep, all summary tables, and all figures. Regenerates everything under
results/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time

from . import graphs, plots
from .experiment import (
    DEGREES,
    GENERATOR_VALIDATION_DEGREES,
    GENERATOR_VALIDATION_N,
    N_GRID,
    cell_summary_to_row,
    compare_generators,
    fit_gap_power_law,
    run_sweep,
    summarize_sweep,
    trial_result_to_row,
)
from .spectrum import extremal_eigenvalues
from .theory import alon_boppana_bound

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def run_exact_validation() -> list[dict]:
    """Cross-check the eigensolver against three graphs whose spectra are
    exactly known by hand: K_{d+1}, K_{d,d}, and the Petersen graph."""
    rows = []

    for d in (3, 5, 8):
        adj = graphs.complete_graph_adjacency(d)
        spec = extremal_eigenvalues(adj, d)
        exact_l2_abs = 1.0  # spectrum {d, -1 x d}: lambda2=-1, |lambda_min|=-1 -> max=1
        rows.append(
            {
                "name": f"K_{{{d+1}}}",
                "d": d,
                "computed_lambda1": spec.lambda1,
                "exact_lambda1": float(d),
                "computed_lambda2_abs": spec.lambda2_abs,
                "exact_lambda2_abs": exact_l2_abs,
            }
        )

    for d in (3, 5, 8):
        adj = graphs.complete_bipartite_regular_adjacency(d)
        spec = extremal_eigenvalues(adj, d)
        exact_l2_abs = float(d)  # spectrum {d, -d, 0 x (2d-2)}: lambda(G) = d
        rows.append(
            {
                "name": f"K_{{{d},{d}}}",
                "d": d,
                "computed_lambda1": spec.lambda1,
                "exact_lambda1": float(d),
                "computed_lambda2_abs": spec.lambda2_abs,
                "exact_lambda2_abs": exact_l2_abs,
            }
        )

    adj = graphs.petersen_graph_adjacency()
    spec = extremal_eigenvalues(adj, 3)
    rows.append(
        {
            "name": "Petersen",
            "d": 3,
            "computed_lambda1": spec.lambda1,
            "exact_lambda1": 3.0,
            "computed_lambda2_abs": spec.lambda2_abs,
            "exact_lambda2_abs": 2.0,  # spectrum {3, 1 x5, -2 x4}
        }
    )

    return rows


def write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("Running exact-spectrum validation (K_{d+1}, K_{d,d}, Petersen)...")
    validation_rows = run_exact_validation()
    max_validation_err = max(
        abs(r["computed_lambda1"] - r["exact_lambda1"]) for r in validation_rows
    )
    max_validation_err = max(
        max_validation_err,
        max(abs(r["computed_lambda2_abs"] - r["exact_lambda2_abs"]) for r in validation_rows),
    )
    write_csv(validation_rows, os.path.join(RESULTS_DIR, "exact_validation.csv"))
    plots.plot_exact_validation(validation_rows, os.path.join(RESULTS_DIR, "exact_validation.png"))
    print(f"  max |computed - exact| eigenvalue error: {max_validation_err:.2e}")

    print("Cross-validating the from-scratch pairing-model generator against networkx...")
    generator_comparisons = [
        compare_generators(d, n) for d in GENERATOR_VALIDATION_DEGREES for n in GENERATOR_VALIDATION_N
    ]
    write_csv(generator_comparisons, os.path.join(RESULTS_DIR, "generator_cross_validation.csv"))
    max_disagreement_se = max(abs(c["diff_in_standard_errors"]) for c in generator_comparisons)
    print(f"  max |mean difference| in pooled standard errors: {max_disagreement_se:.2f}")

    print(f"Running sweep over d in {DEGREES}, n in {N_GRID}...")
    results = run_sweep(degrees=DEGREES, n_grid=N_GRID)
    print(f"  {len(results)} random d-regular graphs generated and diagonalized")

    cells = summarize_sweep(results)
    fits = {d: fit_gap_power_law(cells, d) for d in DEGREES}

    trial_rows = [trial_result_to_row(r) for r in results]
    write_csv(trial_rows, os.path.join(RESULTS_DIR, "trials.csv"))

    cell_rows = [cell_summary_to_row(c) for c in cells]
    write_csv(cell_rows, os.path.join(RESULTS_DIR, "summary.csv"))

    n_disconnected = sum(1 for r in results if not r.connected)
    n_bipartite_like = sum(1 for r in results if r.bipartite_like)
    # Alon-Boppana is an asymptotic (liminf) statement, not a per-instance
    # guarantee -- individual graphs, especially at moderate n, routinely
    # sit *below* 2*sqrt(d-1) (see README/gap_convergence_loglog.png). That
    # is expected, not a violation of the theorem. What would be a genuine
    # surprise is a graph *exceeding* the bound by a large, non-shrinking
    # margin as n grows; we track that separately.
    n_below_bound = sum(1 for r in results if r.gap < -1e-6)
    n_exceeds_bound = sum(1 for r in results if r.gap > 1e-6)
    worst_below_bound = min((r.gap for r in results), default=0.0)
    worst_exceeds_bound = max((r.gap for r in results), default=0.0)

    elapsed = time.time() - t0

    summary = {
        "degrees": list(DEGREES),
        "n_grid": list(N_GRID),
        "total_graphs": len(results),
        "alon_boppana_bounds": {str(d): alon_boppana_bound(d) for d in DEGREES},
        "exact_validation": {
            "max_abs_error": max_validation_err,
            "rows": validation_rows,
        },
        "generator_cross_validation": {
            "max_disagreement_in_standard_errors": max_disagreement_se,
            "rows": generator_comparisons,
        },
        "n_disconnected": n_disconnected,
        "n_bipartite_like": n_bipartite_like,
        "n_below_bound": n_below_bound,
        "worst_below_bound_gap": worst_below_bound,
        "n_exceeds_bound": n_exceeds_bound,
        "worst_exceeds_bound_gap": worst_exceeds_bound,
        "power_law_fits": fits,
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Generating figures...")
    plots.plot_lambda2_vs_n(cells, DEGREES, os.path.join(RESULTS_DIR, "lambda2_vs_n.png"))
    plots.plot_gap_loglog(cells, fits, DEGREES, os.path.join(RESULTS_DIR, "gap_convergence_loglog.png"))
    plots.plot_within_eps_fraction(cells, DEGREES, os.path.join(RESULTS_DIR, "near_ramanujan_fraction.png"))
    plots.plot_distribution_comparison(results, DEGREES, os.path.join(RESULTS_DIR, "distribution_comparison.png"))

    print(f"Done in {elapsed:.1f}s. disconnected={n_disconnected}, "
          f"bipartite-like={n_bipartite_like}, below bound={n_below_bound} "
          f"(worst={worst_below_bound:.4f}), exceeds bound={n_exceeds_bound} "
          f"(worst={worst_exceeds_bound:.4f})")


if __name__ == "__main__":
    main()
