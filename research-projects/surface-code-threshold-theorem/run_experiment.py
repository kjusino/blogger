#!/usr/bin/env python3
"""Entry point: sweep the rotated surface code over code distance and
physical error rate, decode with MWPM, and test the threshold theorem's
two predictions:

  1. Existence of a threshold p_th where logical-error curves for
     different code distances cross (higher distance helps below p_th,
     hurts above it).
  2. Sub-threshold power-law scaling P_L(d, p) ~ p^floor((d+1)/2).

Usage:
    python run_experiment.py                       # default settings
    python run_experiment.py --shots 5000 --distances 3 5   # quick smoke test
"""

import argparse
import csv
import json
import os

import numpy as np

from src.experiment import run_sweep, summarize
from src.plots import generate_all_figures

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9])
    parser.add_argument("--p-min", type=float, default=0.001)
    parser.add_argument("--p-max", type=float, default=0.02)
    parser.add_argument("--num-p", type=int, default=18)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    parser.add_argument("--fig-dir", type=str, default=os.path.join(HERE, "figures"))
    return parser.parse_args()


def write_csv(summary: dict, path: str) -> None:
    fieldnames = [
        "distance", "p", "shots", "logical_errors",
        "logical_error_rate", "ci_lo", "ci_hi",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["raw_results"]:
            writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    for d in args.distances:
        if d < 3 or d % 2 == 0:
            raise ValueError(f"distance must be an odd integer >= 3, got {d}")

    p_values = np.geomspace(args.p_min, args.p_max, args.num_p)

    print(f"Sweeping distances={args.distances}, {args.num_p} p-values in "
          f"[{args.p_min}, {args.p_max}], {args.shots} shots each "
          f"({len(args.distances) * args.num_p} total configurations)...")

    results = run_sweep(args.distances, p_values, shots=args.shots, base_seed=args.seed)
    summary = summarize(p_values, results)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    csv_path = os.path.join(args.out_dir, "raw_results.csv")
    write_csv(summary, csv_path)
    print(f"Wrote {csv_path}")

    generate_all_figures(summary, args.fig_dir)
    print(f"Wrote figures to {args.fig_dir}")

    if summary["threshold_estimate"] is not None:
        print(f"\nEstimated threshold p_th = {summary['threshold_estimate']:.5f}")
    else:
        print("\nNo threshold crossing found in the swept range.")

    print("\nSub-threshold scaling exponents (fitted vs. predicted):")
    for d in summary["distances"]:
        fit = summary["subthreshold_exponent_fits"][str(d)]
        if "slope" in fit:
            print(
                f"  d={d}: fitted={fit['slope']:.2f} +/- {fit['slope_stderr']:.2f}, "
                f"predicted={fit['predicted_slope']}, r={fit['r_value']:.4f}"
            )
        else:
            print(f"  d={d}: could not fit ({fit['error']})")


if __name__ == "__main__":
    main()
