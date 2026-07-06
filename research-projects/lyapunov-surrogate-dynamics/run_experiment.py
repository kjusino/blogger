#!/usr/bin/env python3
"""CLI entry point for the "Do Learned Surrogate Models Preserve Chaos?"
experiment: sweeps {train_size} x {noise_level} x {hidden_width}, trains a
from-scratch-NumPy MLP surrogate of the Lorenz-63 flow map for each config,
measures one-step validation MSE, the learned leading Lyapunov exponent
(vs. the true value), chaos-detection accuracy, and an attractor-shape
divergence metric -- then saves a results table and four summary figures.

Usage:
    python3 run_experiment.py            # full sweep (a few minutes on CPU)
    python3 run_experiment.py --quick    # tiny grid, a few epochs, seconds
"""
from __future__ import annotations

import argparse
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from src.experiment import (
    GridSpec, quick_grid_spec, run_grid, train_representative_surrogate,
    TRUE_LAMBDA1_LITERATURE,
)
from src.plotting import generate_all_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                         help="Run a drastically reduced grid for a fast "
                              "integration-test-style smoke run.")
    parser.add_argument("--no-figures", action="store_true",
                         help="Skip figure generation (results table only).")
    parser.add_argument("--results-dir", default=os.path.join(THIS_DIR, "results"))
    parser.add_argument("--figures-dir", default=os.path.join(THIS_DIR, "figures"))
    args = parser.parse_args()

    spec = quick_grid_spec() if args.quick else GridSpec()

    print("=" * 70)
    print("Do Learned Surrogate Models Preserve Chaos?")
    print(f"Mode: {'QUICK (smoke test)' if args.quick else 'FULL'}")
    print(f"Grid: train_sizes={spec.train_sizes} noise_levels={spec.noise_levels} "
          f"hidden_widths={spec.hidden_widths} seeds={spec.seeds}")
    print("=" * 70)

    rows = run_grid(spec, out_dir=args.results_dir, verbose=True, quick=args.quick)

    n_correct = sum(1 for r in rows if r["chaos_detected_correct"])
    chaos_accuracy = n_correct / len(rows) if rows else float("nan")

    summary = {
        "n_configs": len(rows),
        "chaos_detection_accuracy": chaos_accuracy,
        "true_lambda1_literature_reference": TRUE_LAMBDA1_LITERATURE,
        "true_lambda1_measured": rows[0]["lambda1_true"] if rows else None,
    }

    if not args.no_figures:
        print("\nGenerating figures (retraining one representative surrogate "
              "for the attractor-comparison figure)...")
        # A representative "reasonably well-trained" config for the
        # true-vs-surrogate attractor figure: largest training set, no
        # observation noise, a moderate hidden width.
        rep_train_size = max(spec.train_sizes)
        rep_hidden_width = sorted(spec.hidden_widths)[len(spec.hidden_widths) // 2]
        representative = train_representative_surrogate(
            train_size=rep_train_size, noise_level=0.0,
            hidden_width=rep_hidden_width, seed=spec.seeds[0], spec=spec,
        )
        fig_summary = generate_all_figures(rows, args.figures_dir,
                                            representative=representative)
        summary.update(fig_summary)

    summary_path = os.path.join(args.results_dir, "summary.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nResults table: {os.path.join(args.results_dir, 'grid_results.csv')}")
    if not args.no_figures:
        print(f"Figures: {args.figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
