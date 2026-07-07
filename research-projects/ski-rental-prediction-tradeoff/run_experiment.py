#!/usr/bin/env python3
"""CLI entry point for the ski-rental-with-predictions experiment.

Usage:
    python3 run_experiment.py            # full experiment grid
    python3 run_experiment.py --quick    # fast smoke-test grid (seconds)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_full_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="run a fast smoke-test grid instead of the full experiment"
    )
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    figures_dir = os.path.join(here, "figures")

    summary = run_full_experiment(results_dir, figures_dir, quick=args.quick)

    print()
    print("=" * 72)
    print(f"Done in {summary['elapsed_seconds']:.2f}s (quick_mode={summary['quick_mode']})")
    print("=" * 72)
    print(json.dumps(summary["classical_bound_check"], indent=2))
    print()
    print("Theory vs brute force:")
    print(json.dumps(summary["theory_vs_bruteforce"], indent=2))
    print()
    print("Monte Carlo lambda*(sigma):")
    print(json.dumps(summary["monte_carlo"]["lambda_star_empirical_by_sigma"], indent=2))
    print(f"heuristic c = {summary['monte_carlo']['lambda_star_heuristic_c']:.4f}")
    print(f"lambda* monotonic non-increasing in sigma: {summary['monte_carlo']['lambda_star_monotonic_nonincreasing']}")
    print()
    print(f"Full results written to {results_dir}")
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
