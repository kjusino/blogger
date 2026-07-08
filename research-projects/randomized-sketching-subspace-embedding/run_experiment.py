#!/usr/bin/env python3
"""CLI entry point for the randomized-sketching / subspace-embedding experiment.

Usage:
    python run_experiment.py            # full production sweep (~1-2 min)
    python run_experiment.py --smoke    # fast sanity check (~seconds)
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import FULL_CONFIG, SMOKE_CONFIG, run_all
from src.plots import make_all_plots

HERE = os.path.dirname(os.path.abspath(__file__))


def flatten_results(results):
    rows = []
    for sweep in ["threshold", "scaling", "coherence", "least_squares", "timing_vs_k", "timing_vs_n"]:
        for r in results[sweep]:
            row = dict(sweep=sweep)
            row.update(r)
            rows.append(row)
    return rows


def write_csv(rows, path):
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def crossover_k(rows, sketch, target=0.5):
    """Smallest k at which success_rate >= target for a given sketch, else None."""
    rs = sorted((r for r in rows if r["sketch"] == sketch), key=lambda r: r["k"])
    for r in rs:
        if r["success_rate"] >= target:
            return r["k"]
    return None


def build_summary(results, fit_info):
    threshold_rows = results["threshold"]
    coherence_rows = results["coherence"]

    def coherence_max(basis, variant):
        vals = [r["max_eps"] for r in coherence_rows if r["basis"] == basis and r["variant"] == variant]
        return max(vals) if vals else None

    return dict(
        config=results["config"],
        k0_threshold=results["k0_threshold"],
        k0_least_squares=results["k0_least_squares"],
        threshold_crossover_k={
            name: crossover_k(threshold_rows, name) for name in ["gaussian", "srht", "countsketch"]
        },
        scaling_fits={name: dict(a=a, exponent=b, r2=r2) for name, (a, b, r2) in fit_info["scaling_fits"].items()},
        timing_fits={name: dict(a=a, exponent=b, r2=r2) for name, (a, b, r2) in fit_info["timing_fits"].items()},
        coherence_ablation=dict(
            incoherent_srht_precond_max_eps=coherence_max("incoherent", "srht_precond"),
            incoherent_uniform_sampling_max_eps=coherence_max("incoherent", "uniform_sampling"),
            coherent_srht_precond_max_eps=coherence_max("coherent", "srht_precond"),
            coherent_uniform_sampling_max_eps=coherence_max("coherent", "uniform_sampling"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="fast sanity-check sweep instead of the full run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=os.path.join(HERE, "results"))
    parser.add_argument("--figdir", default=os.path.join(HERE, "figures"))
    args = parser.parse_args()

    cfg = SMOKE_CONFIG if args.smoke else FULL_CONFIG
    print(f"Running {'SMOKE' if args.smoke else 'FULL'} sweep (seed={args.seed})...")

    results = run_all(cfg, args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    rows = flatten_results(results)
    write_csv(rows, os.path.join(args.outdir, "raw_results.csv"))
    print(f"Wrote {len(rows)} rows to {os.path.join(args.outdir, 'raw_results.csv')}")

    fit_info = make_all_plots(results, args.figdir)
    print(f"Wrote 6 figures to {args.figdir}")

    summary = build_summary(results, fit_info)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {os.path.join(args.outdir, 'summary.json')}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
