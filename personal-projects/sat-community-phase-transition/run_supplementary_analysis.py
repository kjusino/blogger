#!/usr/bin/env python3
"""Two follow-up checks that explain the main sweep's headline result
(community structure at mu=0.0 shifts the SAT/UNSAT transition to a LOWER
alpha and *raises* the search-effort peak, rather than lowering it).

1. Weakest-link / independence check. At mu=0.0 every clause is local, so
   the formula is a disjoint union of n_communities independent smaller
   random-3-SAT instances (no shared variables). If that's really what's
   driving the shifted transition, then P(SAT) of the whole formula should
   match p_component(alpha)^k, where p_component is the satisfiability
   probability of a single bare random-3-SAT instance of the community's
   size. This simulates p_component directly and compares the prediction
   to the mu=0.0 curve already in results/results.csv.

2. Decomposition-efficiency check. A solver that exploited the disjoint
   variable structure could solve each community separately and sum the
   work; ours (plain DPLL, no component detection) does not. This measures
   how much extra search the monolithic solver burns relative to solving
   the same instance's parts independently, at alphas around the mu=0.0
   hardness peak identified in the main sweep.

Requires results/results.csv (run run_experiment.py first).
"""

import csv
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analysis import aggregate_by_alpha_mu, load_rows, sorted_unique
from src.cnf import community_3sat, decompose_by_community, random_3sat
from src.solver import solve

HERE = os.path.dirname(os.path.abspath(__file__))
N_VARS_TOTAL = 80
N_COMMUNITIES = 4
N_VARS_COMPONENT = N_VARS_TOTAL // N_COMMUNITIES
COMPONENT_TRIALS = 300
DECOMPOSITION_TRIALS = 30
DECISION_CAP = 2_000_000


def independence_check(alphas, mu0_p_sat_by_alpha, seed=100):
    rng = random.Random(seed)
    rows = []
    for alpha in alphas:
        m = round(alpha * N_VARS_COMPONENT)
        sat_count = 0
        for _ in range(COMPONENT_TRIALS):
            cnf = random_3sat(n_vars=N_VARS_COMPONENT, n_clauses=m, rng=rng)
            if solve(cnf, decision_cap=DECISION_CAP).satisfiable:
                sat_count += 1
        p_component = sat_count / COMPONENT_TRIALS
        predicted = p_component ** N_COMMUNITIES
        observed = mu0_p_sat_by_alpha[alpha]
        rows.append({
            "alpha": alpha,
            "p_sat_component": p_component,
            "predicted_p_sat_whole": predicted,
            "observed_p_sat_mu0": observed,
            "abs_error": abs(predicted - observed),
        })
    return rows


def decomposition_check(alphas, trials=DECOMPOSITION_TRIALS, seed=200):
    rng = random.Random(seed)
    rows = []
    for alpha in alphas:
        m = round(alpha * N_VARS_TOTAL)
        for trial in range(trials):
            cnf = community_3sat(
                n_vars=N_VARS_TOTAL, n_clauses=m, n_communities=N_COMMUNITIES,
                mu=0.0, rng=rng,
            )
            mono = solve(cnf, decision_cap=DECISION_CAP)
            parts = decompose_by_community(cnf)
            part_results = [solve(p, decision_cap=DECISION_CAP) for p in parts]

            decisions_sum_parts = sum(r.decisions for r in part_results)
            satisfiable_parts = all(r.satisfiable for r in part_results)
            assert mono.satisfiable == satisfiable_parts, (
                "monolithic and decomposed solves disagree on satisfiability"
            )

            rows.append({
                "alpha": alpha,
                "trial": trial,
                "decisions_monolithic": mono.decisions,
                "decisions_sum_parts": decisions_sum_parts,
                "overhead_ratio": (mono.decisions / decisions_sum_parts) if decisions_sum_parts > 0 else float("nan"),
            })
    return rows


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    results_path = os.path.join(HERE, "results", "results.csv")
    if not os.path.exists(results_path):
        print("results/results.csv not found -- run run_experiment.py first", file=sys.stderr)
        sys.exit(1)

    main_rows = load_rows(results_path)
    alphas = sorted_unique(main_rows, "alpha")
    agg = aggregate_by_alpha_mu(main_rows)
    mu0_p_sat_by_alpha = {a: agg[(a, 0.0)]["p_sat"] for a in alphas}

    print("Running independence check "
          f"({len(alphas)} alphas x {COMPONENT_TRIALS} component trials)...")
    indep_rows = independence_check(alphas, mu0_p_sat_by_alpha)
    mae = statistics.mean(r["abs_error"] for r in indep_rows)
    print(f"  mean absolute error (predicted vs observed P(SAT)): {mae:.4f}")

    # find the mu=0.0 hardness peak from the main sweep to focus the
    # decomposition check where it matters most.
    mu0_alphas_by_hardness = sorted(
        alphas, key=lambda a: agg[(a, 0.0)]["median_decisions"], reverse=True
    )
    peak_alpha = mu0_alphas_by_hardness[0]
    peak_idx = alphas.index(peak_alpha)
    window = [alphas[i] for i in range(max(0, peak_idx - 1), min(len(alphas), peak_idx + 2))]
    print(f"mu=0.0 hardness peak at alpha={peak_alpha}; "
          f"running decomposition check at alphas={window}")

    decomp_rows = decomposition_check(window)
    overhead_ratios = [r["overhead_ratio"] for r in decomp_rows if r["overhead_ratio"] == r["overhead_ratio"]]
    median_overhead = statistics.median(overhead_ratios)
    print(f"  median overhead ratio (monolithic decisions / sum of decomposed decisions): "
          f"{median_overhead:.2f}x")

    results_dir = os.path.join(HERE, "results")
    write_csv(indep_rows, os.path.join(results_dir, "independence_check.csv"),
              ["alpha", "p_sat_component", "predicted_p_sat_whole", "observed_p_sat_mu0", "abs_error"])
    write_csv(decomp_rows, os.path.join(results_dir, "decomposition_check.csv"),
              ["alpha", "trial", "decisions_monolithic", "decisions_sum_parts", "overhead_ratio"])

    from src.plots import plot_independence_check, plot_decomposition_overhead
    figures_dir = os.path.join(HERE, "figures")
    plot_independence_check(indep_rows, os.path.join(figures_dir, "fig5_independence_check.png"))
    plot_decomposition_overhead(decomp_rows, os.path.join(figures_dir, "fig6_decomposition_overhead.png"))
    print("Wrote fig5_independence_check.png and fig6_decomposition_overhead.png")

    print(f"\nSummary for README:")
    print(f"  independence-model mean absolute error: {mae:.4f}")
    print(f"  median decomposition overhead ratio at peak: {median_overhead:.2f}x")
    print(f"  peak alpha (mu=0.0): {peak_alpha}")


if __name__ == "__main__":
    main()
