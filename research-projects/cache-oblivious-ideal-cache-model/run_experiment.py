#!/usr/bin/env python3
"""
CLI entry point for the ideal-cache matmul study.

Full production sweep:
    python run_experiment.py

Fast smoke test (tiny sizes, for CI / development):
    python run_experiment.py --smoke
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import (
    run_sweep_B,
    run_sweep_M,
    run_sweep_n,
    run_sweep_naive_capacity_cliff,
    run_sweep_tall_cache_boundary,
)
from src.plots import (
    plot_fit_summary,
    plot_naive_capacity_cliff,
    plot_scaling_n,
    plot_scaling_param,
    plot_tall_cache_boundary,
)
from src.theory import PREDICTED_EXPONENT, leading_constant

ALGORITHMS = ("naive", "blocked", "oblivious")

# E1/E2 keep every algorithm safely inside naive's "capacity-sufficient"
# regime (M >= n * B) *and* blocked/oblivious's bandwidth-bound regime
# (n >> sqrt(M), else the working set nearly fits in cache and misses
# are compulsory-miss dominated, ~ n^2/B rather than ~ n^3/(B*sqrt(M))),
# so a single n^3 (resp. B^-1) power law applies to all three algorithms
# and can be compared apples-to-apples.
# E3 tests blocked/oblivious's Theta(1/sqrt(M)) prediction with M staying
# well clear of the tall-cache boundary (M >> B^2). Naive's own M-scaling
# is a step function (see naive_cliff_*), tested separately in E3b.
PRODUCTION = {
    "n_values": [80, 96, 112, 128, 160, 192],
    "n_fixed_B": 8,
    "n_fixed_M": 2048,  # sqrt(M)=45 << n_min=80; n_max*B=1536 << M=2048
    "B_values": [2, 4, 6, 8, 12, 16, 24, 32],
    "B_fixed_n": 64,
    "B_fixed_M": 4096,  # n * B_max = 64*32=2048 -> 2x headroom
    "M_values": [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
    "M_fixed_n": 64,
    "M_fixed_B": 8,  # B^2=64, so M range gives >=8x tall-cache headroom
    "naive_cliff_M_values": [128, 192, 256, 320, 384, 448, 480, 512, 544, 576, 640, 768, 1024],
    "naive_cliff_n": 64,
    "naive_cliff_B": 8,  # threshold M = n*B = 512, bracketed densely above
    "tall_M_values": [128, 256, 512, 768, 1024, 1536, 2048, 4096, 8192],
    "tall_fixed_n": 64,
    "tall_fixed_B": 32,
}

SMOKE = {
    "n_values": [4, 8, 12, 16],
    "n_fixed_B": 4,
    "n_fixed_M": 256,
    "B_values": [2, 4, 8],
    "B_fixed_n": 16,
    "B_fixed_M": 256,
    "M_values": [128, 192, 256, 384, 512],
    "M_fixed_n": 16,
    "M_fixed_B": 4,
    "naive_cliff_M_values": [16, 32, 48, 64, 96, 128],
    "naive_cliff_n": 16,
    "naive_cliff_B": 4,
    "tall_M_values": [16, 32, 64, 128],
    "tall_fixed_n": 16,
    "tall_fixed_B": 8,
}


def parse_args():
    p = argparse.ArgumentParser(description="Ideal-cache matmul study")
    p.add_argument("--smoke", action="store_true", help="run a tiny fast sweep instead of the production sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SMOKE if args.smoke else PRODUCTION
    tag = "[smoke] " if args.smoke else ""

    figures_dir = os.path.join(args.output_dir, "figures")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"{tag}E1 scaling_n: n={cfg['n_values']}, B={cfg['n_fixed_B']}, M={cfg['n_fixed_M']}")
    e1 = run_sweep_n(cfg["n_values"], cfg["n_fixed_B"], cfg["n_fixed_M"], ALGORITHMS, seed=args.seed)

    print(f"{tag}E2 scaling_B: B={cfg['B_values']}, n={cfg['B_fixed_n']}, M={cfg['B_fixed_M']}")
    e2 = run_sweep_B(cfg["B_fixed_n"], cfg["B_values"], cfg["B_fixed_M"], ALGORITHMS, seed=args.seed)

    print(f"{tag}E3 scaling_M (blocked/oblivious): M={cfg['M_values']}, n={cfg['M_fixed_n']}, B={cfg['M_fixed_B']}")
    e3 = run_sweep_M(cfg["M_fixed_n"], cfg["M_fixed_B"], cfg["M_values"], ALGORITHMS, seed=args.seed)

    print(
        f"{tag}E3b naive_capacity_cliff: M={cfg['naive_cliff_M_values']}, "
        f"n={cfg['naive_cliff_n']}, B={cfg['naive_cliff_B']} "
        f"(threshold n*B={cfg['naive_cliff_n']*cfg['naive_cliff_B']})"
    )
    e3b = run_sweep_naive_capacity_cliff(
        cfg["naive_cliff_n"], cfg["naive_cliff_B"], cfg["naive_cliff_M_values"], seed=args.seed
    )

    print(
        f"{tag}E4 tall_cache_boundary: M={cfg['tall_M_values']}, "
        f"n={cfg['tall_fixed_n']}, B={cfg['tall_fixed_B']} (B^2={cfg['tall_fixed_B']**2})"
    )
    e4 = run_sweep_tall_cache_boundary(
        cfg["tall_fixed_n"], cfg["tall_fixed_B"], cfg["tall_M_values"], ALGORITHMS, seed=args.seed
    )

    all_records = e1 + e2 + e3 + e3b + e4

    raw_csv = os.path.join(results_dir, "raw_results.csv")
    with open(raw_csv, "w", newline="") as f:
        fieldnames = [
            "experiment", "algorithm", "n", "B", "M", "tile", "misses", "hits",
            "total_accesses", "seconds", "tall_cache_ratio", "capacity_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    print(f"wrote {raw_csv} ({len(all_records)} rows)")

    fits_n = plot_scaling_n(e1, os.path.join(figures_dir, "scaling_n.png"))
    fits_B = plot_scaling_param(
        e2, "scaling_B", "B", "block size B (words)",
        {a: PREDICTED_EXPONENT[("B", a)] for a in ALGORITHMS},
        os.path.join(figures_dir, "scaling_B.png"),
        "Cache misses vs. block size B (fixed n, M)",
    )
    fits_M = plot_scaling_param(
        e3, "scaling_M", "M", "cache size M (words)",
        {a: PREDICTED_EXPONENT[("M", a)] for a in ("blocked", "oblivious")},
        os.path.join(figures_dir, "scaling_M.png"),
        "Cache misses vs. cache size M, blocked/oblivious (fixed n, B, tall cache)",
        algorithms=("blocked", "oblivious"),
    )
    plot_naive_capacity_cliff(
        e3b, os.path.join(figures_dir, "naive_capacity_cliff.png"),
        cfg["naive_cliff_n"], cfg["naive_cliff_B"],
    )
    plot_tall_cache_boundary(e4, os.path.join(figures_dir, "tall_cache_boundary.png"))

    all_fits = {}
    for algo in ALGORITHMS:
        all_fits[("scaling_n", algo)] = (fits_n[algo], PREDICTED_EXPONENT[("n", algo)])
        all_fits[("scaling_B", algo)] = (fits_B[algo], PREDICTED_EXPONENT[("B", algo)])
    for algo in ("blocked", "oblivious"):
        all_fits[("scaling_M", algo)] = (fits_M[algo], PREDICTED_EXPONENT[("M", algo)])
    plot_fit_summary(all_fits, os.path.join(figures_dir, "fit_summary.png"))
    print(f"wrote figures to {figures_dir}")

    # Leading-constant comparison at the largest n in E1: does oblivious
    # really match blocked's constant, and beat naive's, in the direction
    # theory predicts?
    largest_n = max(cfg["n_values"])
    constants = {}
    for algo in ALGORITHMS:
        row = next(r for r in e1 if r["algorithm"] == algo and r["n"] == largest_n)
        constants[algo] = leading_constant(
            row["n"], row["B"], row["M"], row["misses"], algo
        )

    below = next(r for r in e3b if r["M"] < cfg["naive_cliff_n"] * cfg["naive_cliff_B"])
    above = next(
        r for r in sorted(e3b, key=lambda r: -r["M"])
        if r["M"] >= cfg["naive_cliff_n"] * cfg["naive_cliff_B"]
    )

    summary = {
        "config": {"smoke": args.smoke, "seed": args.seed, **cfg},
        "fits": {
            f"{exp}/{algo}": {
                "slope": fit.slope,
                "predicted_slope": pred,
                "stderr": fit.stderr,
                "r_squared": fit.r_squared,
                "n_points": fit.n_points,
                "within_0.3": fit.within(pred, 0.3),
            }
            for (exp, algo), (fit, pred) in all_fits.items()
        },
        "leading_constants_at_n_max": {"n": largest_n, **constants},
        "naive_over_oblivious_miss_ratio_at_n_max": (
            next(r["misses"] for r in e1 if r["algorithm"] == "naive" and r["n"] == largest_n)
            / next(r["misses"] for r in e1 if r["algorithm"] == "oblivious" and r["n"] == largest_n)
        ),
        "blocked_over_oblivious_miss_ratio_at_n_max": (
            next(r["misses"] for r in e1 if r["algorithm"] == "blocked" and r["n"] == largest_n)
            / next(r["misses"] for r in e1 if r["algorithm"] == "oblivious" and r["n"] == largest_n)
        ),
        "naive_capacity_cliff": {
            "threshold_M": cfg["naive_cliff_n"] * cfg["naive_cliff_B"],
            "misses_just_below_threshold": below["misses"],
            "misses_just_above_threshold": above["misses"],
            "drop_factor": below["misses"] / above["misses"],
            "predicted_drop_factor": cfg["naive_cliff_B"],
        },
    }
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")

    print("\n--- fit summary ---")
    for key, v in summary["fits"].items():
        flag = "OK" if v["within_0.3"] else "MISS"
        print(
            f"{key:24s} fitted={v['slope']:+.3f}  predicted={v['predicted_slope']:+.3f}  "
            f"r2={v['r_squared']:.3f}  [{flag}]"
        )
    print(
        f"\nat n={largest_n}: naive/oblivious miss ratio = "
        f"{summary['naive_over_oblivious_miss_ratio_at_n_max']:.1f}x, "
        f"blocked/oblivious = {summary['blocked_over_oblivious_miss_ratio_at_n_max']:.2f}x"
    )
    cliff = summary["naive_capacity_cliff"]
    print(
        f"naive capacity cliff at M={cliff['threshold_M']}: "
        f"misses drop {cliff['drop_factor']:.1f}x crossing the threshold "
        f"(predicted ~{cliff['predicted_drop_factor']}x)"
    )


if __name__ == "__main__":
    main()
