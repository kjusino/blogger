#!/usr/bin/env python3
"""Single reproducible entry point for the epidemic-threshold experiment.

Builds RR/ER/BA networks at matched mean degree across three sizes, runs the
QS-method discrete-time SIS sweep on each, locates the empirical epidemic
threshold via susceptibility-peak interpolation, compares it against the QMF
(1/lambda_max) and HMF (<k>/<k^2>) theoretical predictions, tests whether the
QMF-vs-HMF accuracy gap tracks degree heterogeneity, and writes all raw
results to results/*.csv and figures to figures/*.png.

Usage:
    python run_experiment.py
"""

from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np

from src.experiment import (
    DEFAULT_NS,
    DEFAULT_TOPOLOGIES,
    MEAN_DEGREE,
    DELTA,
    N_STEPS,
    BURN_IN,
    N_REPEATS,
    N_REALIZATIONS,
    run_full_experiment,
    aggregate_by_topology_n,
    compute_heterogeneity_correlation,
    write_dicts_csv,
)
from src.networks import build_network
from src.plotting import (
    plot_degree_distributions,
    plot_susceptibility_grid,
    plot_threshold_comparison,
    plot_heterogeneity_vs_gap,
)

SEED = 12345
ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def main() -> None:
    t0 = time.time()
    print(f"=== Epidemic-threshold-networks experiment (seed={SEED}) ===")
    print(f"topologies={DEFAULT_TOPOLOGIES} ns={DEFAULT_NS} mean_degree={MEAN_DEGREE} "
          f"delta={DELTA} n_steps={N_STEPS} burn_in={BURN_IN} n_repeats={N_REPEATS} "
          f"n_realizations={N_REALIZATIONS}")

    print("\n[1/4] Running QS-method SIS sweeps over all (topology, N, realization) combinations...")
    sweep_rows, summaries = run_full_experiment(seed=SEED, verbose=True)
    aggregated = aggregate_by_topology_n(summaries)
    write_dicts_csv(sweep_rows, RESULTS_DIR / "susceptibility_sweep.csv")
    write_dicts_csv(summaries, RESULTS_DIR / "threshold_summary_by_realization.csv")
    write_dicts_csv(aggregated, RESULTS_DIR / "threshold_summary.csv")
    print(f"  wrote {len(sweep_rows)} sweep rows -> results/susceptibility_sweep.csv")
    print(f"  wrote {len(summaries)} per-realization rows -> results/threshold_summary_by_realization.csv")
    print(f"  wrote {len(aggregated)} aggregated rows -> results/threshold_summary.csv "
          f"({time.time() - t0:.1f}s elapsed)")

    for s in aggregated:
        print(f"    {s['topology']:>3} N={s['n']:<5} het={s['heterogeneity_ratio']:.3f} "
              f"tau_c_emp={s['tau_c_empirical']:.4f}+/-{s['tau_c_sem']:.4f} "
              f"qmf={s['qmf_threshold']:.4f} (eps={s['eps_qmf']:.3f}) "
              f"hmf={s['hmf_threshold']:.4f} (eps={s['eps_hmf']:.3f})")

    print("\n[2/4] Testing heterogeneity-vs-accuracy-gap correlation (core hypothesis, "
          "all realizations)...")
    correlation = compute_heterogeneity_correlation(summaries)
    write_dicts_csv([correlation], RESULTS_DIR / "heterogeneity_correlation.csv")
    print(f"  n={correlation['n_points']} Spearman rho={correlation['spearman_rho']:.4f} "
          f"exact p={correlation['spearman_p']:.5g}")

    print("\n[3/4] Sampling degree distributions for the largest N...")
    n_show = max(DEFAULT_NS)
    degree_samples = {}
    for topo in DEFAULT_TOPOLOGIES:
        G = build_network(topo, n_show, MEAN_DEGREE, seed=SEED)
        degree_samples[topo] = np.array([d for _, d in G.degree()])

    print("\n[4/4] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    representative_rows = [r for r in sweep_rows if r["realization"] == 0]
    plot_degree_distributions(degree_samples, FIGURES_DIR / "degree_distributions.png")
    plot_susceptibility_grid(representative_rows, aggregated, DEFAULT_NS, DEFAULT_TOPOLOGIES,
                              FIGURES_DIR / "susceptibility_grid.png")
    plot_threshold_comparison(aggregated, FIGURES_DIR / "threshold_comparison.png")
    plot_heterogeneity_vs_gap(summaries, FIGURES_DIR / "heterogeneity_vs_gap.png")
    print("  wrote figures/degree_distributions.png")
    print("  wrote figures/susceptibility_grid.png")
    print("  wrote figures/threshold_comparison.png")
    print("  wrote figures/heterogeneity_vs_gap.png")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
