#!/usr/bin/env python3
"""Run the full collision-tester-vs-naive-learner sample-complexity grid,
write results/*.csv + results/summary.json, and render the figures used in
the README.

Usage:
    python3 run_experiment.py            # full grid (~1-2 minutes)
    python3 run_experiment.py --quick     # small smoke-test grid (seconds)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.experiment import find_m50, fit_power_law, power_curve
from src.theory import PREDICTED_EXPONENT, paired_collision_probability

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

TESTER_LABELS = {"collision": "collision tester", "naive_learner": "naive learner"}
FAMILY_LABELS = {
    "paired": "paired (worst-case)",
    "single_heavy": "single heavy element",
    "block_quarter": "block (n/4 elements)",
}


def run_scaling_sweep(n_grids, epsilons, trials, rng):
    """Part A: for each tester, sweep n at fixed family='paired', find m50(n),
    and fit the empirical scaling exponent."""
    rows = []
    fits = {}
    for tester_name, n_grid in n_grids.items():
        for epsilon in epsilons:
            m50s = []
            for n in n_grid:
                res = find_m50(tester_name, "paired", n, epsilon, trials, rng)
                rows.append({
                    "tester": tester_name,
                    "family": "paired",
                    "epsilon": epsilon,
                    "n": n,
                    "m50": res.m50,
                    "false_positive_rate": res.false_positive_rate,
                    "bracketed": res.bracketed,
                })
                m50s.append(res.m50)
                if not res.bracketed:
                    print(
                        f"  [warn] m50 not bracketed within search grid for "
                        f"tester={tester_name} n={n} eps={epsilon}; using "
                        f"boundary value {res.m50:.1f}"
                    )
            slope, intercept, r2 = fit_power_law(n_grid, m50s)
            fits[f"{tester_name}_eps{epsilon}"] = {
                "tester": tester_name,
                "epsilon": epsilon,
                "fitted_exponent": slope,
                "intercept": intercept,
                "r_squared": r2,
                "predicted_exponent": PREDICTED_EXPONENT[tester_name],
            }
            print(
                f"  {TESTER_LABELS[tester_name]:16s} eps={epsilon:.2f}  "
                f"fitted slope={slope:.3f} (predicted {PREDICTED_EXPONENT[tester_name]})  "
                f"R^2={r2:.4f}"
            )
    return rows, fits


def run_family_comparison(n, epsilon, m_multipliers, trials, rng):
    """Part B: at fixed (n, epsilon), compare the collision tester's power
    curve across distribution families that are all exactly epsilon-far in
    TV distance but structurally different."""
    from src.theory import collision_tester_predicted_m

    center = collision_tester_predicted_m(n, epsilon)
    m_grid = sorted({max(2, int(round(center * mult))) for mult in m_multipliers})

    rows = []
    for family in ["paired", "single_heavy", "block_quarter"]:
        powers = power_curve("collision", family, n, epsilon, m_grid, trials, rng)
        for m, power in zip(m_grid, powers):
            rows.append({"family": family, "m": m, "power": power})
    return rows, m_grid


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(scaling_rows, fits, tester_name, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    epsilons = sorted({r["epsilon"] for r in scaling_rows if r["tester"] == tester_name})
    colors = plt.cm.viridis(np.linspace(0.15, 0.75, len(epsilons)))
    for color, epsilon in zip(colors, epsilons):
        pts = [r for r in scaling_rows if r["tester"] == tester_name and r["epsilon"] == epsilon]
        pts.sort(key=lambda r: r["n"])
        ns = [r["n"] for r in pts]
        m50s = [r["m50"] for r in pts]
        ax.loglog(ns, m50s, "o-", color=color, label=f"eps={epsilon}")
        fit = fits[f"{tester_name}_eps{epsilon}"]
        n_line = np.array([min(ns), max(ns)], dtype=float)
        m_line = np.exp(fit["intercept"]) * n_line ** fit["fitted_exponent"]
        ax.loglog(n_line, m_line, "--", color=color, alpha=0.6,
                   label=f"fit slope={fit['fitted_exponent']:.2f}")
    ax.set_xlabel("domain size n")
    ax.set_ylabel("empirical m50 (samples for 50% detection power)")
    ax.set_title(f"{TESTER_LABELS[tester_name]}: m50(n) vs n (family=paired)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(scaling_rows, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    epsilon = sorted({r["epsilon"] for r in scaling_rows})[0]
    coll = {r["n"]: r["m50"] for r in scaling_rows if r["tester"] == "collision" and r["epsilon"] == epsilon}
    naive = {r["n"]: r["m50"] for r in scaling_rows if r["tester"] == "naive_learner" and r["epsilon"] == epsilon}
    common_ns = sorted(set(coll) & set(naive))

    ax1.loglog(list(coll.keys()), list(coll.values()), "o-", label="collision tester")
    ax1.loglog(list(naive.keys()), list(naive.values()), "s-", label="naive learner")
    ax1.set_xlabel("domain size n")
    ax1.set_ylabel("empirical m50")
    ax1.set_title(f"Sample complexity comparison (eps={epsilon})")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    ratios = [naive[n] / coll[n] for n in common_ns]
    ax2.loglog(common_ns, ratios, "o-", color="darkred")
    ax2.set_xlabel("domain size n")
    ax2.set_ylabel("m50(naive learner) / m50(collision tester)")
    ax2.set_title("Sublinear-tester speedup factor vs n")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_family_curves(family_rows, m_grid, n, epsilon, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for family in ["paired", "single_heavy", "block_quarter"]:
        pts = [r for r in family_rows if r["family"] == family]
        pts.sort(key=lambda r: r["m"])
        ax.semilogx([p["m"] for p in pts], [p["power"] for p in pts], "o-",
                    label=FAMILY_LABELS[family])
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("number of samples m")
    ax.set_ylabel("detection power")
    ax.set_title(f"Collision tester power by distribution family\n(n={n}, eps={epsilon}, all exactly TV={epsilon}-far)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_collision_probability_gap(n, epsilons, out_path):
    """Sanity-check figure: the exact collision-probability gap the tester's
    threshold is calibrated against, as a function of epsilon."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    eps_range = np.linspace(0, 0.5, 200)
    null_prob = np.full_like(eps_range, 1.0 / n)
    alt_prob = np.array([paired_collision_probability(n, e) for e in eps_range])
    ax.plot(eps_range, null_prob * n, label="uniform: n * sum p_i^2 = 1")
    ax.plot(eps_range, alt_prob * n, label="paired (eps-far): n * sum p_i^2 = 1+4*eps^2")
    ax.plot(eps_range, (1 + 2 * eps_range ** 2), "--", color="gray",
             label="tester threshold: n * sum p_i^2 = 1+2*eps^2")
    for e in epsilons:
        ax.axvline(e, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("epsilon (TV distance from uniform)")
    ax.set_ylabel("n * collision probability")
    ax.set_title(f"Collision-probability gap used by the tester (n={n})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a small smoke-test grid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.quick:
        n_grids = {"collision": [64, 128, 256], "naive_learner": [64, 128, 256]}
        epsilons = [0.2]
        trials = 20
        fam_n, fam_eps, fam_trials = 128, 0.2, 20
        m_multipliers = [0.25, 0.5, 1, 2, 4]
    else:
        n_grids = {
            "collision": [64, 128, 256, 512, 1024, 2048],
            "naive_learner": [64, 128, 256, 512, 1024],
        }
        epsilons = [0.15, 0.25]
        trials = 150
        fam_n, fam_eps, fam_trials = 512, 0.2, 200
        m_multipliers = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]

    print("=== Part A: fitting m50(n) scaling exponents (family=paired) ===")
    scaling_rows, fits = run_scaling_sweep(n_grids, epsilons, trials, rng)
    write_csv(scaling_rows, RESULTS_DIR / "scaling_results.csv")

    print("\n=== Part B: distribution-family comparison at fixed (n, epsilon) ===")
    family_rows, m_grid = run_family_comparison(fam_n, fam_eps, m_multipliers, fam_trials, rng)
    write_csv(family_rows, RESULTS_DIR / "family_power_curves.csv")

    summary = {
        "fits": fits,
        "family_comparison": {"n": fam_n, "epsilon": fam_eps, "m_grid": m_grid},
        "config": {
            "quick": args.quick,
            "n_grids": n_grids,
            "epsilons": epsilons,
            "trials": trials,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Rendering figures ===")
    plot_scaling(scaling_rows, fits, "collision", FIGURES_DIR / "collision_scaling.png")
    plot_scaling(scaling_rows, fits, "naive_learner", FIGURES_DIR / "naive_learner_scaling.png")
    plot_comparison(scaling_rows, FIGURES_DIR / "scaling_comparison.png")
    plot_family_curves(family_rows, m_grid, fam_n, fam_eps, FIGURES_DIR / "family_power_curves.png")
    plot_collision_probability_gap(fam_n, epsilons, FIGURES_DIR / "collision_probability_gap.png")

    print(f"\nWrote results to {RESULTS_DIR} and figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
