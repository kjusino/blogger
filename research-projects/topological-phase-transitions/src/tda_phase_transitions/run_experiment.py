"""CLI entry point: run the full sweep, save results.csv, summary.json and plots.

Usage: python -m tda_phase_transitions.run_experiment
"""

from __future__ import annotations

import os
import time

from . import experiment, plots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")

N_VALUES = (50, 100, 200, 400)
TRIALS = 30


def main() -> None:
    results_dir = os.path.abspath(RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()

    def progress(msg: str) -> None:
        print(f"[{time.time() - t0:6.1f}s] {msg}")

    print("Running parameter sweep...")
    results = experiment.run_sweep(
        models=list(experiment.MODEL_NAMES),
        n_values=N_VALUES,
        trials=TRIALS,
        progress=progress,
    )

    experiment.save_results_csv(results, os.path.join(results_dir, "results.csv"))
    summary = experiment.save_summary_json(results, os.path.join(results_dir, "summary.json"))
    print("Summary:", summary)

    print("Generating plots...")
    max_n = max(N_VALUES)
    for model in experiment.MODEL_NAMES:
        plots.plot_model_curves(
            model, max_n, seed=999, out_path=os.path.join(results_dir, f"{model}_curves.png")
        )
        plots.plot_finite_size_collapse(
            model, list(N_VALUES), seed_base=5000,
            out_path=os.path.join(results_dir, f"{model}_finite_size_collapse.png"),
        )

    plots.plot_threshold_convergence(results, os.path.join(results_dir, "threshold_convergence.png"))
    plots.plot_cycle_vs_percolation(results, os.path.join(results_dir, "cycle_vs_percolation.png"))

    print(f"Done in {time.time() - t0:.1f}s. Results in {results_dir}")


if __name__ == "__main__":
    main()
