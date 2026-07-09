#!/usr/bin/env python3
"""Run the curvature-vs-greedy-ratio experiment end to end.

Usage:
    python3 run_experiment.py            # full production sweep
    python3 run_experiment.py --quick    # fast smoke test (small grid)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src.experiment import compute_summary, run_sweep
from src.plots import generate_all_figures

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                         help="run a small grid for a fast sanity check")
    args = parser.parse_args()

    if args.quick:
        records = run_sweep(
            n_values=(8, 12),
            redundancy_mults=(0.0, 0.2, 1.0, 5.0, 20.0),
            seeds_per_config=3,
            k_fracs=(0.3, 0.5),
            base_seed=2026,
        )
    else:
        records = run_sweep()

    summary, per_instance = compute_summary(records)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    pd.DataFrame(per_instance).to_csv(
        os.path.join(RESULTS_DIR, "per_instance_results.csv"), index=False
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    generate_all_figures(records, summary, FIGURES_DIR)

    print(json.dumps(summary, indent=2))
    print()
    print("OVERALL:", "PASS" if summary["overall_pass"] else "FAIL")


if __name__ == "__main__":
    main()
