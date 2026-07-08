#!/usr/bin/env python3
"""Entry point: run the full graphical-balanced-allocations sweep,
fit growth models, generate figures, and write results/summary.json
with the concrete success-criteria checks from the README.

Usage: python run_experiment.py [--trials N] [--seed N]
Runtime: ~1-3 minutes on a single core.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from src import experiment, fitting, graphs, plotting

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

EXPANDER_KEYS = ["complete", "regular3", "regular10", "erdos_renyi", "smallworld_high_rewiring"]
POOR_KEYS = ["smallworld_low_rewiring", "torus", "cycle", "path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=experiment.DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    def progress(family, n):
        print(f"  ran {family:28s} n={n}", flush=True)

    print(f"Running sweep: families={len(graphs.FAMILIES)}, ns={experiment.DEFAULT_NS}, trials={args.trials}")
    raw_df, summary_df = experiment.run_sweep(trials=args.trials, seed=args.seed, progress_cb=progress)

    raw_df.to_csv(RESULTS_DIR / "raw_trials.csv", index=False)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    # --- growth model fits, one per graph family (baselines excluded: they
    # are reference constants, not families whose growth regime we're testing)
    fit_rows = []
    for family in graphs.FAMILIES:
        rows = summary_df[summary_df["family"] == family].sort_values("n")
        try:
            fit = fitting.fit_growth_models(rows["n"].to_numpy(), rows["mean_gap"].to_numpy())
        except Exception as exc:  # pragma: no cover - defensive, not expected to trigger
            print(f"  fit failed for {family}: {exc}")
            continue
        fit_rows.append(
            {
                "family": family,
                "log_r2": fit.log_r2,
                "loglog_r2": fit.loglog_r2,
                "best_model": fit.best_model,
                "log_params": fit.log_params,
                "loglog_params": fit.loglog_params,
            }
        )
    import pandas as pd

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(RESULTS_DIR / "growth_fits.csv", index=False)

    # --- spectral gap vs max-load gap correlation, across all graph-family runs
    graph_rows = summary_df[summary_df["family"].isin(graphs.FAMILIES)]
    rho, pval = stats.spearmanr(graph_rows["mean_spectral_gap"], graph_rows["mean_gap"])

    # --- figures
    plotting.plot_max_load_gap(summary_df, FIGURES_DIR / "max_load_gap.png")
    plotting.plot_spectral_gap(summary_df, FIGURES_DIR / "spectral_gap.png")
    plotting.plot_gap_vs_spectral_gap(summary_df, FIGURES_DIR / "gap_vs_spectral_gap.png", rho, pval)
    largest_n = summary_df["n"].max()
    plotting.plot_ranking_bar(summary_df, FIGURES_DIR / "ranking_bar.png", largest_n)

    # --- success criteria, evaluated concretely against the README claims
    largest_n_rows = summary_df[summary_df["requested_n"] == experiment.DEFAULT_NS[-1]]

    def gap_at_largest(family):
        r = largest_n_rows[largest_n_rows["family"] == family]
        return float(r["mean_gap"].iloc[0]) if not r.empty else None

    expander_gaps = {f: gap_at_largest(f) for f in EXPANDER_KEYS}
    poor_gaps = {f: gap_at_largest(f) for f in POOR_KEYS}
    one_choice_gap = gap_at_largest("one_choice")
    classical_two_choice_gap = gap_at_largest("classical_two_choice")

    expander_loglog_wins = sum(
        1 for f in EXPANDER_KEYS if fit_df.loc[fit_df["family"] == f, "best_model"].iloc[0] == "loglog"
    )
    poor_log_wins = sum(
        1 for f in POOR_KEYS if fit_df.loc[fit_df["family"] == f, "best_model"].iloc[0] == "log"
    )

    expanders_below_baseline = sum(
        1 for v in expander_gaps.values() if v is not None and classical_two_choice_gap is not None and v <= classical_two_choice_gap + 2
    )
    poor_exceed_expanders = all(
        poor_gaps[p] is not None and expander_gaps["complete"] is not None and poor_gaps[p] > expander_gaps["complete"]
        for p in POOR_KEYS
        if poor_gaps[p] is not None
    )

    summary = {
        "trials": args.trials,
        "ns": list(experiment.DEFAULT_NS),
        "largest_n": int(largest_n),
        "spearman_rho_spectral_gap_vs_max_load_gap": float(rho),
        "spearman_p_value": float(pval),
        "expander_gaps_at_largest_n": expander_gaps,
        "poor_expander_gaps_at_largest_n": poor_gaps,
        "one_choice_gap_at_largest_n": one_choice_gap,
        "classical_two_choice_gap_at_largest_n": classical_two_choice_gap,
        "growth_fits": fit_df.set_index("family")[["best_model", "log_r2", "loglog_r2"]].to_dict("index"),
        "criteria": {
            "spectral_gap_negatively_correlated_with_max_load_gap": bool(rho < -0.4 and pval < 0.05),
            "majority_of_expanders_fit_loglog_growth": bool(expander_loglog_wins >= 3),
            "majority_of_poor_expanders_fit_log_growth": bool(poor_log_wins >= 2),
            "expanders_track_classical_two_choice_baseline": bool(expanders_below_baseline >= 3),
            "poor_expanders_all_exceed_complete_graph": bool(poor_exceed_expanders),
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Success criteria ===")
    for key, value in summary["criteria"].items():
        print(f"  [{'PASS' if value else 'FAIL'}] {key}")
    print(f"\nSpearman rho={rho:.3f}, p={pval:.2g}")
    print(f"Wrote results/ and figures/ under {ROOT}")


if __name__ == "__main__":
    main()
