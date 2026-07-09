#!/usr/bin/env python3
"""Run the full EXP3-vs-UCB1 regret-scaling experiment grid.

    python3 run_experiment.py            # full grid (~1-2 min)
    python3 run_experiment.py --quick    # smoke-test grid (a few seconds)

Writes results/*.csv + results/summary.json and figures/*.png.
"""

import argparse
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiment import algorithm_comparison, exp3_scaling_over_K, exp3_scaling_over_T
from src.regret import exp3_bound, fit_log_law, fit_power_law, ucb1_bound

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

DELTA = 0.3
K_MAIN = 8


def write_csv(path, records):
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def aggregate_by(records, key):
    grouped = {}
    for r in records:
        grouped.setdefault(r[key], []).append(r["regret"])
    xs = sorted(grouped)
    means = [float(np.mean(grouped[x])) for x in xs]
    stds = [float(np.std(grouped[x])) for x in xs]
    return xs, means, stds


def run_exp3_scaling_over_T(quick):
    t_values = [500, 1000, 2000] if quick else [500, 1000, 2000, 4000, 8000, 16000, 32000]
    n_seeds = 3 if quick else 15
    records = exp3_scaling_over_T(n_arms=K_MAIN, t_values=t_values, delta=DELTA, n_seeds=n_seeds, base_seed=1)
    write_csv(os.path.join(RESULTS_DIR, "exp3_scaling_over_T.csv"), records)

    t_sorted, mean_regret, std_regret = aggregate_by(records, "T")
    exponent, r_squared = fit_power_law(t_sorted, mean_regret)
    bounds = [exp3_bound(K_MAIN, t) for t in t_sorted]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(t_sorted, mean_regret, "o-", label="EXP3 measured (mean regret)", color="#1f77b4")
    ax.fill_between(
        t_sorted,
        [m - s for m, s in zip(mean_regret, std_regret)],
        [m + s for m, s in zip(mean_regret, std_regret)],
        alpha=0.2,
        color="#1f77b4",
    )
    ax.loglog(t_sorted, bounds, "--", label="Theoretical bound $2\\sqrt{e-1}\\sqrt{KT\\ln K}$", color="#888888")
    ref = mean_regret[0] * (np.array(t_sorted) / t_sorted[0]) ** 0.5
    ax.loglog(t_sorted, ref, ":", label="Reference slope 0.5", color="#d62728")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Final regret")
    ax.set_title(f"EXP3 regret vs T (K={K_MAIN}), fitted exponent={exponent:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_regret_scaling_T.png"), dpi=150)
    plt.close(fig)

    return {
        "t_values": t_sorted,
        "mean_regret": mean_regret,
        "fitted_exponent_T": exponent,
        "r_squared_T": r_squared,
        "n_seeds": n_seeds,
    }


def run_exp3_scaling_over_K(quick):
    k_values = [2, 4, 8] if quick else [2, 4, 8, 16, 32, 64]
    n_seeds = 3 if quick else 15
    horizon = 4000 if quick else 8000
    records = exp3_scaling_over_K(k_values=k_values, horizon=horizon, delta=DELTA, n_seeds=n_seeds, base_seed=2)
    write_csv(os.path.join(RESULTS_DIR, "exp3_scaling_over_K.csv"), records)

    k_sorted, mean_regret, std_regret = aggregate_by(records, "K")
    exponent, r_squared = fit_power_law(k_sorted, mean_regret)
    bounds = [exp3_bound(k, horizon) for k in k_sorted]
    # The bound itself is sqrt(K ln K), not pure sqrt(K): fit its exponent
    # over this same K grid for an apples-to-apples theory comparison,
    # instead of comparing the empirical fit to a naive 0.5.
    bound_exponent, _ = fit_power_law(k_sorted, bounds)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.loglog(k_sorted, mean_regret, "o-", label="EXP3 measured (mean regret)", color="#1f77b4")
    ax.fill_between(
        k_sorted,
        [m - s for m, s in zip(mean_regret, std_regret)],
        [m + s for m, s in zip(mean_regret, std_regret)],
        alpha=0.2,
        color="#1f77b4",
    )
    ax.loglog(k_sorted, bounds, "--", label="Theoretical bound $2\\sqrt{e-1}\\sqrt{KT\\ln K}$", color="#888888")
    ref = mean_regret[0] * (np.array(k_sorted) / k_sorted[0]) ** 0.5
    ax.loglog(k_sorted, ref, ":", label="Reference slope 0.5", color="#d62728")
    ax.set_xlabel("Number of arms K")
    ax.set_ylabel("Final regret")
    ax.set_title(f"EXP3 regret vs K (T={horizon}), fitted exponent={exponent:.3f} (bound: {bound_exponent:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "exp3_regret_scaling_K.png"), dpi=150)
    plt.close(fig)

    return {
        "k_values": k_sorted,
        "mean_regret": mean_regret,
        "fitted_exponent_K": exponent,
        "bound_exponent_K": bound_exponent,
        "r_squared_K": r_squared,
        "horizon": horizon,
        "n_seeds": n_seeds,
    }


def run_stochastic_comparison(quick):
    t_values = [500, 1000, 2000] if quick else [500, 1000, 2000, 4000, 8000, 16000, 32000]
    n_seeds = 3 if quick else 15
    records = algorithm_comparison(
        "stochastic", n_arms=K_MAIN, t_values=t_values, delta=DELTA, n_seeds=n_seeds, base_seed=3
    )
    write_csv(os.path.join(RESULTS_DIR, "stochastic_comparison.csv"), records)

    exp3_records = [r for r in records if r["algo"] == "EXP3"]
    ucb1_records = [r for r in records if r["algo"] == "UCB1"]
    t_e, regret_e, _ = aggregate_by(exp3_records, "T")
    t_u, regret_u, _ = aggregate_by(ucb1_records, "T")

    exp3_exponent, exp3_r2 = fit_power_law(t_e, regret_e)
    gaps = [DELTA] * (K_MAIN - 1)
    ucb1_log_coef, ucb1_intercept, ucb1_r2 = fit_log_law(t_u, regret_u)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(t_e, regret_e, "o-", label=f"EXP3 (fitted T^{exp3_exponent:.2f})", color="#1f77b4")
    sign = "+" if ucb1_intercept >= 0 else "-"
    ax.plot(
        t_u, regret_u, "s-",
        label=f"UCB1 (fitted {ucb1_log_coef:.1f}·ln T {sign} {abs(ucb1_intercept):.1f})", color="#2ca02c",
    )
    ax.plot(t_u, [ucb1_bound(gaps, t) for t in t_u], "--", color="#2ca02c", alpha=0.5, label="UCB1 theoretical bound")
    ax.set_xscale("log")
    ax.set_xlabel("Horizon T")
    ax.set_ylabel("Final regret")
    ax.set_title(f"Stationary env (K={K_MAIN}, gap={DELTA}): UCB1 vs EXP3")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "stochastic_ucb1_vs_exp3.png"), dpi=150)
    plt.close(fig)

    return {
        "t_values": t_e,
        "exp3_mean_regret": regret_e,
        "ucb1_mean_regret": regret_u,
        "exp3_fitted_exponent": exp3_exponent,
        "exp3_r_squared": exp3_r2,
        "ucb1_fitted_log_coef": ucb1_log_coef,
        "ucb1_r_squared": ucb1_r2,
        "ucb1_wins_at_max_T": regret_u[-1] < regret_e[-1],
        "n_seeds": n_seeds,
    }


def run_switching_sweep(quick):
    horizon = 4000 if quick else 8000
    num_segments_values = (
        [4, 32, horizon // K_MAIN // 5] if quick else [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, horizon // 5]
    )
    num_segments_values = sorted(set(v for v in num_segments_values if v >= 1))
    n_seeds = 3 if quick else 12
    n_arms = 5

    records = []
    for num_segments in num_segments_values:
        recs = algorithm_comparison(
            "switching",
            n_arms=n_arms,
            t_values=[horizon],
            delta=DELTA,
            n_seeds=n_seeds,
            base_seed=4 * 10_000 + num_segments,
            num_segments=num_segments,
        )
        for r in recs:
            r["num_segments"] = num_segments
            r["segment_length"] = horizon // num_segments
        records.extend(recs)
    write_csv(os.path.join(RESULTS_DIR, "switching_sweep.csv"), records)

    exp3_records = [r for r in records if r["algo"] == "EXP3"]
    ucb1_records = [r for r in records if r["algo"] == "UCB1"]
    seg_e, regret_e, _ = aggregate_by(exp3_records, "segment_length")
    seg_u, regret_u, _ = aggregate_by(ucb1_records, "segment_length")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(seg_e, regret_e, "o-", label="EXP3 (guaranteed-robust)", color="#1f77b4")
    ax.plot(seg_u, regret_u, "s-", label="UCB1 (no robustness guarantee)", color="#2ca02c")
    ax.axvline(n_arms, color="#888888", linestyle=":", alpha=0.7, label=f"segment length = K = {n_arms}")
    ax.set_xscale("log")
    ax.set_xlabel("Segment length (rounds per switch)")
    ax.set_ylabel("Dynamic (per-round-best-arm) regret")
    ax.set_title(f"Switching env (K={n_arms}, T={horizon}, gap={DELTA}):\ndynamic regret vs switching frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "switching_regret_vs_frequency.png"), dpi=150)
    plt.close(fig)

    ucb1_never_worse = all(u <= e * 1.05 for u, e in zip(regret_u, regret_e))

    return {
        "horizon": horizon,
        "n_arms": n_arms,
        "segment_lengths": seg_e,
        "exp3_mean_regret": regret_e,
        "ucb1_mean_regret": regret_u,
        "ucb1_never_worse_than_exp3": ucb1_never_worse,
        "n_seeds": n_seeds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a fast smoke-test grid instead of the full grid")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    t0 = time.time()

    print("[1/4] EXP3 regret scaling vs T (stationary env)...")
    t_summary = run_exp3_scaling_over_T(args.quick)
    print(f"      fitted exponent = {t_summary['fitted_exponent_T']:.3f} (theory: 0.5), R^2={t_summary['r_squared_T']:.4f}")

    print("[2/4] EXP3 regret scaling vs K (stationary env)...")
    k_summary = run_exp3_scaling_over_K(args.quick)
    print(
        f"      fitted exponent = {k_summary['fitted_exponent_K']:.3f} "
        f"(theory over this same K grid, sqrt(K ln K): {k_summary['bound_exponent_K']:.3f}), "
        f"R^2={k_summary['r_squared_K']:.4f}"
    )

    print("[3/4] UCB1 vs EXP3 on stationary env...")
    stoch_summary = run_stochastic_comparison(args.quick)
    print(f"      UCB1 wins at max T: {stoch_summary['ucb1_wins_at_max_T']}")

    print("[4/4] UCB1 vs EXP3 dynamic regret across switching frequencies...")
    switch_summary = run_switching_sweep(args.quick)
    print(f"      UCB1 never worse than EXP3 (5% slack): {switch_summary['ucb1_never_worse_than_exp3']}")

    # Combined summary figure: fitted exponents + crossover behavior
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    labels = ["vs T", "vs K"]
    measured = [t_summary["fitted_exponent_T"], k_summary["fitted_exponent_K"]]
    theory = [0.5, k_summary["bound_exponent_K"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, measured, width, label="measured", color="#1f77b4")
    ax.bar(x + width / 2, theory, width, label="theory (bound, same grid)", color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fitted power-law exponent")
    ax.set_title("EXP3 scaling-law exponents")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    seg = switch_summary["segment_lengths"]
    ax.plot(seg, switch_summary["exp3_mean_regret"], "o-", label="EXP3", color="#1f77b4")
    ax.plot(seg, switch_summary["ucb1_mean_regret"], "s-", label="UCB1", color="#2ca02c")
    ax.set_xscale("log")
    ax.set_xlabel("Segment length")
    ax.set_ylabel("Dynamic regret")
    ax.set_title("Price of robustness under switching")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "summary.png"), dpi=150)
    plt.close(fig)

    summary = {
        "delta": DELTA,
        "exp3_scaling_over_T": t_summary,
        "exp3_scaling_over_K": k_summary,
        "stochastic_comparison": stoch_summary,
        "switching_sweep": switch_summary,
        "wall_clock_seconds": time.time() - t0,
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {summary['wall_clock_seconds']:.1f}s. Results in results/, figures in figures/.")


if __name__ == "__main__":
    main()
