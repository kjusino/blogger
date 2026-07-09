#!/usr/bin/env python3
"""Run the full Sinkhorn / Birkhoff-contraction-rate experiment sweep and
save results (CSV + summary JSON) to results/, plus figures to figures/.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.experiment import (
    records_to_dicts,
    run_cost_convergence_check,
    run_extreme_sweep,
    run_main_sweep,
)
from src.plotting import make_all_figures

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    print("Running main sweep (n=m=30, 4 families x epsilon grid)...")
    main_records = run_main_sweep()
    print(f"  {len(main_records)} records in {time.time() - t0:.1f}s")

    t1 = time.time()
    print("Running extreme sweep (small eps, bound-looseness demonstration)...")
    extreme_records = run_extreme_sweep()
    print(f"  {len(extreme_records)} records in {time.time() - t1:.1f}s")

    t2 = time.time()
    print("Running entropic-cost-vs-exact-OT convergence check...")
    cost_points = run_cost_convergence_check()
    print(f"  {len(cost_points)} points in {time.time() - t2:.1f}s")

    main_rows = records_to_dicts(main_records)
    extreme_rows = records_to_dicts(extreme_records)
    cost_rows = [
        {"eps": p.eps, "entropic_cost": p.entropic_cost, "exact_cost": p.exact_cost, "gap": p.gap}
        for p in cost_points
    ]

    write_csv(main_rows, RESULTS_DIR / "main_sweep.csv")
    write_csv(extreme_rows, RESULTS_DIR / "extreme_sweep.csv")
    write_csv(cost_rows, RESULTS_DIR / "cost_convergence.csv")

    # --- Summary statistics -------------------------------------------------
    violations = [r for r in main_records if r.bound_violated]
    fittable = [r for r in main_records if r.tightness is not None]

    def tightness_stats(records):
        vals = np.array([r.tightness for r in records])
        if len(vals) == 0:
            return {"n": 0}
        return {
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    families = sorted(set(r.family for r in main_records))
    tightness_by_family = {
        fam: tightness_stats([r for r in fittable if r.family == fam]) for fam in families
    }

    structured = [r for r in fittable if r.family in ("clustered_points", "grid_1d")]
    unstructured = [r for r in fittable if r.family == "iid_random"]

    # Marginal (unmatched) tightness medians above are confounded: families
    # that converge more slowly at large eps have fittable-rate data over a
    # *wider* eps range (including the high-eps, high-tightness regime),
    # which inflates their aggregate median independent of any real
    # structural effect. The controlled comparison holds eps fixed and
    # compares families only at eps values where *all four* have fittable
    # data.
    eps_sets = {
        fam: set(round(r.eps, 6) for r in fittable if r.family == fam) for fam in families
    }
    common_eps = sorted(set.intersection(*eps_sets.values())) if eps_sets else []
    matched_eps_medians = {}
    for eps in common_eps:
        row = {}
        for fam in families:
            vals = [r.tightness for r in fittable if r.family == fam and round(r.eps, 6) == eps]
            row[fam] = float(np.median(vals)) if vals else None
        matched_eps_medians[str(eps)] = row

    extreme_by_family = {}
    for fam in sorted(set(r.family for r in extreme_records)):
        fam_records = [r for r in extreme_records if r.family == fam]
        extreme_by_family[fam] = {
            "eps": sorted(set(round(r.eps, 6) for r in fam_records)),
            "mean_n_iter_by_eps": {
                str(round(eps, 6)): float(
                    np.mean([r.n_iter for r in fam_records if round(r.eps, 6) == round(eps, 6)])
                )
                for eps in sorted(set(r.eps for r in fam_records))
            },
        }

    summary = {
        "n_main_records": len(main_records),
        "n_extreme_records": len(extreme_records),
        "n_bound_violations": len(violations),
        "n_fittable_rate_records": len(fittable),
        "tightness_by_family": tightness_by_family,
        "tightness_structured_median_unmatched": float(
            np.median([r.tightness for r in structured])
        ) if structured else None,
        "tightness_unstructured_median_unmatched": float(
            np.median([r.tightness for r in unstructured])
        ) if unstructured else None,
        "tightness_by_family_at_matched_eps": matched_eps_medians,
        "extreme_sweep_mean_iters_by_family_eps": extreme_by_family,
        "cost_convergence_final_gap": cost_rows[-1]["gap"] if cost_rows else None,
        "cost_convergence_initial_gap": cost_rows[0]["gap"] if cost_rows else None,
        "wallclock_seconds": time.time() - t0,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    print("Generating figures...")
    make_all_figures(main_records, extreme_records, cost_points, FIGURES_DIR)
    print(f"Done in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
