"""Reproducible experiment suite for the quorum-staleness-pbs project.

Runs three experiment groups:

1. Validation: homogeneous replicas + uniformly-random read quorum selection
   (i.e. the WARS model's own assumptions, reproduced in the simulator).
   The simulated staleness curve should match the theoretical WARS curve
   within Monte Carlo noise, across several (N, W, R, latency family) configs.

2. Divergence example: one config with heterogeneous replica speeds and a
   *fixed* read quorum pinned to the slowest replicas (a sticky/regional
   client). Plots the empirical curve clearly diverging above the
   theoretical curve.

3. Divergence sweep: the same config, sweeping the heterogeneity parameter
   sigma_het from 0 (homogeneous) upward, showing the theory/empirical gap
   grow monotonically with replica-speed heterogeneity.

All randomness is seeded explicitly, so re-running this script reproduces
bit-identical results.json and plots.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pbs.comparison import area_between, monte_carlo_noise_floor, sup_distance  # noqa: E402
from pbs.simulator import simulate_staleness_curve  # noqa: E402
from pbs.wars_model import wars_staleness_curve  # noqa: E402

PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"

N_TRIALS = 150_000
DELTAS = np.linspace(0.0, 5.0, 21)


def run_validation_group() -> list[dict]:
    configs = [
        dict(label="N=5,W=1,R=1,Exp", n_replicas=5, w=1, r=1, family="exponential", rate=1.0),
        dict(label="N=5,W=2,R=2,Exp", n_replicas=5, w=2, r=2, family="exponential", rate=1.0),
        dict(label="N=7,W=2,R=2,LogNormal", n_replicas=7, w=2, r=2, family="lognormal", rate=1.0, sigma=1.0),
    ]
    noise_floor = monte_carlo_noise_floor(N_TRIALS)
    results = []
    for i, cfg in enumerate(configs):
        label = cfg.pop("label")
        theory = wars_staleness_curve(
            deltas=DELTAS, n_trials=N_TRIALS, rng=np.random.default_rng(1000 + i), **cfg,
        )
        empirical = simulate_staleness_curve(
            deltas=DELTAS, n_trials=N_TRIALS, rng=np.random.default_rng(2000 + i),
            sigma_het=0.0, selection="random", **cfg,
        ).curve
        dist = sup_distance(theory, empirical)
        area = area_between(theory, empirical, DELTAS)
        results.append(
            dict(
                label=label,
                config=cfg,
                theory_curve=theory.tolist(),
                empirical_curve=empirical.tolist(),
                sup_distance=dist,
                area_between=area,
                noise_floor=noise_floor,
                matches_theory=bool(dist < noise_floor * 2.5),
            )
        )
        print(f"[validation] {label}: sup_distance={dist:.4f} (noise floor {noise_floor:.4f})")
    return results


def run_divergence_example() -> dict:
    cfg = dict(n_replicas=6, w=2, r=2, family="exponential", rate=1.0)
    sigma_het = 1.5
    theory = wars_staleness_curve(deltas=DELTAS, n_trials=N_TRIALS, rng=np.random.default_rng(3000), **cfg)
    empirical = simulate_staleness_curve(
        deltas=DELTAS, n_trials=N_TRIALS, rng=np.random.default_rng(4000),
        sigma_het=sigma_het, selection="fixed", **cfg,
    ).curve
    dist = sup_distance(theory, empirical)
    area = area_between(theory, empirical, DELTAS)
    print(f"[divergence example] sigma_het={sigma_het}: sup_distance={dist:.4f}, area={area:.4f}")
    return dict(
        config=cfg,
        sigma_het=sigma_het,
        theory_curve=theory.tolist(),
        empirical_curve=empirical.tolist(),
        sup_distance=dist,
        area_between=area,
    )


def run_divergence_sweep() -> dict:
    """Sweep replica-speed heterogeneity, averaging over independent replica-speed
    realizations at each level (not just one draw of multipliers), so the trend
    reflects the *expected* effect of heterogeneity rather than single-instance
    sampling noise in which particular replicas happen to be slow.
    """
    cfg = dict(n_replicas=6, w=2, r=2, family="exponential", rate=1.0)
    sigma_het_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    reps_per_level = 12
    sweep_trials = 60_000
    theory = wars_staleness_curve(deltas=DELTAS, n_trials=N_TRIALS, rng=np.random.default_rng(5000), **cfg)

    sup_distances = []
    areas = []
    for i, sigma_het in enumerate(sigma_het_values):
        rep_curves = []
        for rep in range(reps_per_level):
            seed = 6000 + i * reps_per_level + rep
            empirical = simulate_staleness_curve(
                deltas=DELTAS, n_trials=sweep_trials, rng=np.random.default_rng(seed),
                sigma_het=sigma_het, selection="fixed", **cfg,
            ).curve
            rep_curves.append(empirical)
        avg_curve = np.mean(rep_curves, axis=0)
        sup_distances.append(sup_distance(theory, avg_curve))
        areas.append(area_between(theory, avg_curve, DELTAS))
        print(f"[sweep] sigma_het={sigma_het:.2f}: sup_distance={sup_distances[-1]:.4f} (avg of {reps_per_level} reps)")

    return dict(
        config=cfg,
        sigma_het_values=sigma_het_values,
        reps_per_level=reps_per_level,
        sweep_trials=sweep_trials,
        sup_distances=sup_distances,
        areas=areas,
        noise_floor=monte_carlo_noise_floor(sweep_trials),
    )


def plot_validation(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, r in zip(axes, results):
        ax.plot(DELTAS, r["theory_curve"], "--", color="tab:blue", label="WARS theory")
        ax.plot(DELTAS, r["empirical_curve"], "-", color="tab:orange", alpha=0.8, label="Simulator")
        ax.set_title(r["label"], fontsize=10)
        ax.set_xlabel("delta (time since commit)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("P(read is stale)")
    axes[0].legend()
    fig.suptitle("Validation: homogeneous replicas + random quorum selection matches WARS theory")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "validation_curves.png", dpi=150)
    plt.close(fig)


def plot_divergence_example(result: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(DELTAS, result["theory_curve"], "--", color="tab:blue", label="WARS theory (i.i.d. assumption)")
    ax.plot(DELTAS, result["empirical_curve"], "-", color="tab:red", label="Simulator (heterogeneous + sticky reads)")
    ax.fill_between(DELTAS, result["theory_curve"], result["empirical_curve"], color="tab:red", alpha=0.15)
    ax.set_xlabel("delta (time since commit)")
    ax.set_ylabel("P(read is stale)")
    ax.set_title(f"Divergence example (sigma_het={result['sigma_het']}):\ntheory understates real staleness", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "divergence_curve_example.png", dpi=150)
    plt.close(fig)


def plot_divergence_sweep(sweep: dict) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax1.plot(sweep["sigma_het_values"], sweep["sup_distances"], "o-", color="tab:red", label="sup-distance (KS-style)")
    ax1.axhline(sweep["noise_floor"], color="gray", linestyle=":", label="Monte Carlo noise floor")
    ax1.set_xlabel("replica speed heterogeneity (sigma_het)")
    ax1.set_ylabel("sup-distance from WARS theory")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(sweep["sigma_het_values"], sweep["areas"], "s--", color="tab:purple", alpha=0.7, label="area between curves")
    ax2.set_ylabel("area between curves")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax1.set_title("Model divergence grows with replica-speed heterogeneity\n(fixed sticky-slow read quorum)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "divergence_vs_heterogeneity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    start = time.time()
    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    validation_results = run_validation_group()
    divergence_example = run_divergence_example()
    divergence_sweep = run_divergence_sweep()

    plot_validation(validation_results)
    plot_divergence_example(divergence_example)
    plot_divergence_sweep(divergence_sweep)

    elapsed = time.time() - start

    all_validation_pass = all(r["matches_theory"] for r in validation_results)
    monotonic_increase = all(
        divergence_sweep["sup_distances"][i] <= divergence_sweep["sup_distances"][i + 1] + 1e-6
        for i in range(len(divergence_sweep["sup_distances"]) - 1)
    )

    results = dict(
        n_trials=N_TRIALS,
        deltas=DELTAS.tolist(),
        validation=validation_results,
        divergence_example=divergence_example,
        divergence_sweep=divergence_sweep,
        success_metrics=dict(
            all_validation_configs_match_theory=all_validation_pass,
            divergence_sweep_monotonic_nondecreasing=monotonic_increase,
            divergence_example_sup_distance_exceeds_noise_floor=bool(
                divergence_example["sup_distance"] > divergence_sweep["noise_floor"] * 5
            ),
        ),
        elapsed_seconds=elapsed,
    )

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Wrote results/results.json and 3 plots to plots/.")
    print(f"Success metrics: {results['success_metrics']}")


if __name__ == "__main__":
    main()
