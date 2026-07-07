"""End-to-end experiment: does empirical double-descent risk match the
closed-form theory for isotropic min-norm linear regression?

Runs a gamma = p/n sweep at fixed n, a bias/variance decomposition check,
and a secondary ridge-regularization sweep; writes JSON results to
`results/` and figures to `figures/`; prints a pass/fail verdict against
the success metrics documented in the README.
"""

import json
import os

import numpy as np

from src.experiment import run_grid, run_ridge_sweep
from src.plotting import (
    plot_risk_vs_gamma,
    plot_relative_error,
    plot_bias_variance_decomposition,
    plot_ridge_sweep,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIGURES_DIR = os.path.join(HERE, "figures")

# Model parameters
N = 200
R2 = 1.0
SIGMA2 = 1.0
N_TRIALS = 300
SEED = 20260707

GAMMAS = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96, 0.99,
    1.01, 1.04, 1.07, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
    5.0,
]

RELATIVE_ERROR_THRESHOLD = 0.08
CORRELATION_THRESHOLD = 0.999

RIDGE_GAMMAS = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0]
RIDGE_LAMBDAS = [0.0, 0.1, 1.0, 5.0, 20.0]
RIDGE_TRIALS = 200


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Running main gamma sweep: n={N}, {len(GAMMAS)} gamma values, "
          f"{N_TRIALS} Monte Carlo trials each...")
    results = run_grid(N, GAMMAS, R2, SIGMA2, N_TRIALS, seed=SEED)

    with open(os.path.join(RESULTS_DIR, "grid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Running ridge-regularization sweep...")
    ridge_results = run_ridge_sweep(
        N, RIDGE_GAMMAS, RIDGE_LAMBDAS, R2, SIGMA2, RIDGE_TRIALS,
        seed=SEED + 100000,
    )
    with open(os.path.join(RESULTS_DIR, "ridge_results.json"), "w") as f:
        json.dump(ridge_results, f, indent=2)

    print("Generating figures...")
    plot_risk_vs_gamma(
        results, R2, SIGMA2,
        os.path.join(FIGURES_DIR, "risk_vs_gamma.png"),
    )
    plot_relative_error(
        results, os.path.join(FIGURES_DIR, "relative_error_vs_gamma.png"),
        RELATIVE_ERROR_THRESHOLD,
    )
    plot_bias_variance_decomposition(
        results, os.path.join(FIGURES_DIR, "bias_variance_decomposition.png"),
    )
    plot_ridge_sweep(
        ridge_results, os.path.join(FIGURES_DIR, "ridge_sweep.png"),
    )

    # The exact-risk formula involves E[tr(W^-1)] of a near-singular Wishart
    # matrix. That expectation is finite whenever the "extra" degrees of
    # freedom (n-p-1 or p-n-1) is > 0, but its *variance* is only finite once
    # that same quantity is > 2 -- otherwise the estimator is heavy-tailed
    # (Var[tr(W^-1)] undefined) and no realistic number of Monte Carlo trials
    # converges tightly. Configs that close to the threshold (here: only
    # gamma=0.99 and 1.01, at n=200) are kept in the plots to show the peak,
    # but excluded from the quantitative success metrics for that reason.
    def extra_dof(row):
        return (row["n"] - row["p"] - 1) if row["p"] < row["n"] \
            else (row["p"] - row["n"] - 1)

    core = [r for r in results if extra_dof(r) > 2]
    excluded = [r for r in results if extra_dof(r) <= 2]

    rel_errors = np.array([r["relative_error"] for r in core])
    emp = np.array([r["empirical_risk"] for r in core])
    theory = np.array([r["theoretical_risk"] for r in core])
    correlation = np.corrcoef(emp, theory)[0, 1]

    peak_gamma_empirical = results[int(np.argmax(
        [r["empirical_risk"] for r in results]
    ))]["gamma"]

    summary = {
        "n_configs_total": len(results),
        "n_configs_core": len(core),
        "excluded_near_threshold_gammas": [r["gamma"] for r in excluded],
        "mean_relative_error_core": float(rel_errors.mean()),
        "max_relative_error_core": float(rel_errors.max()),
        "pearson_correlation_empirical_vs_theory_core": float(correlation),
        "peak_gamma_empirical": peak_gamma_empirical,
        "success_mean_relative_error": bool(
            rel_errors.mean() < RELATIVE_ERROR_THRESHOLD
        ),
        "success_correlation": bool(correlation > CORRELATION_THRESHOLD),
        "success_peak_near_threshold": bool(abs(peak_gamma_empirical - 1) < 0.1),
    }
    overall_pass = all(
        v for k, v in summary.items() if k.startswith("success_")
    )
    summary["overall_pass"] = overall_pass

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("OVERALL:", "PASS" if overall_pass else "FAIL")


if __name__ == "__main__":
    main()
