"""Driver: run the construction sweep (RANKING/greedy vs n, adversarial vs
ROM), the search-refinement pass, and the control sweep; save CSV results
and render figures.

Usage: python run_experiment.py
"""

import csv
import json
import os

import numpy as np

from src.experiment import (run_construction_sweep, run_control_sweep,
                             run_search_refinement, staircase_hard_instance,
                             summarize_trend)
from src.search import ranking_ratio_single, ranking_ratios
from src import plotting

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

CONSTRUCTION_N_VALUES = [4, 8, 16, 32, 64, 128, 256, 512]
SEARCH_N_VALUES = [8, 32, 128, 512]
CONTROL_N_VALUES = [16, 64, 256]
CONTROL_P_VALUES = [0.1, 0.3, 0.6]
DISTRIBUTION_N = 64  # representative n for the per-trial histogram
SEED = 20260709

EVAL_TRIALS = 500
CONTROL_TRIALS = 200
SEARCH_ITERATIONS = 80
SEARCH_TRIALS_PER_EVAL = 25


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"Construction sweep over n = {CONSTRUCTION_N_VALUES} ...")
    construction_rows = run_construction_sweep(CONSTRUCTION_N_VALUES, rng, eval_trials=EVAL_TRIALS)
    write_csv(os.path.join(RESULTS_DIR, "construction_sweep.csv"),
              construction_rows, list(construction_rows[0].keys()))

    print(f"Search refinement over n = {SEARCH_N_VALUES} ...")
    search_rows, search_histories = run_search_refinement(
        SEARCH_N_VALUES, rng, n_iterations=SEARCH_ITERATIONS,
        trials_per_eval=SEARCH_TRIALS_PER_EVAL)
    write_csv(os.path.join(RESULTS_DIR, "search_refinement.csv"),
              search_rows, list(search_rows[0].keys()))
    with open(os.path.join(RESULTS_DIR, "search_histories.json"), "w") as f:
        json.dump({str(n): h for n, h in search_histories.items()}, f)

    print(f"Control sweep over n = {CONTROL_N_VALUES}, p = {CONTROL_P_VALUES} ...")
    control_rows = run_control_sweep(CONTROL_N_VALUES, CONTROL_P_VALUES, rng,
                                      eval_trials=CONTROL_TRIALS)
    write_csv(os.path.join(RESULTS_DIR, "control_sweep.csv"),
              control_rows, list(control_rows[0].keys()))

    print(f"Per-trial ratio distribution at n = {DISTRIBUTION_N} ...")
    graph, order = staircase_hard_instance(DISTRIBUTION_N)
    adv_samples = ranking_ratios(graph, order, rng, 4000, opt_size=DISTRIBUTION_N)
    rom_samples = [ranking_ratio_single(graph, rng.permutation(DISTRIBUTION_N), rng, DISTRIBUTION_N)
                   for _ in range(4000)]
    with open(os.path.join(RESULTS_DIR, "distribution_samples.json"), "w") as f:
        json.dump({"n": DISTRIBUTION_N, "adversarial": adv_samples, "rom": rom_samples}, f)

    summary = summarize_trend(construction_rows)
    summary["search_max_improvement"] = max(r["improvement"] for r in search_rows)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))

    print("Rendering figures ...")
    plotting.plot_ratio_vs_n(construction_rows, os.path.join(FIGURES_DIR, "ratio_vs_n.png"))
    plotting.plot_adversarial_vs_rom(construction_rows, os.path.join(FIGURES_DIR, "adversarial_vs_rom.png"))
    plotting.plot_ratio_distribution(adv_samples, rom_samples, DISTRIBUTION_N,
                                      os.path.join(FIGURES_DIR, "ratio_distribution.png"))
    representative_search_n = SEARCH_N_VALUES[len(SEARCH_N_VALUES) // 2]
    plotting.plot_search_convergence(search_histories[representative_search_n],
                                      representative_search_n,
                                      os.path.join(FIGURES_DIR, "search_convergence.png"))
    plotting.plot_control_vs_adversarial(control_rows, construction_rows,
                                          os.path.join(FIGURES_DIR, "control_vs_adversarial.png"))

    print("Done. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
