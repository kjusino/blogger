"""Monte Carlo sweeps comparing empirical risk to the closed-form theory."""

import numpy as np

from .data import generate_beta0, sample_dataset
from .estimators import fit_min_norm, fit_ridge
from .theory import exact_risk, bias_variance_exact


def run_config(n: int, p: int, r2: float, sigma2: float, n_trials: int,
                seed: int):
    """Fit the min-norm interpolator on `n_trials` independent datasets.

    Returns the per-trial risks alongside an empirical bias^2/variance
    decomposition (computed from the trial-to-trial mean and spread of
    beta_hat), so each theoretical component can be checked separately,
    not just their sum.
    """
    rng = np.random.default_rng(seed)
    beta0 = generate_beta0(p, r2)
    beta_hats = np.empty((n_trials, p))
    for t in range(n_trials):
        X, y = sample_dataset(n, p, beta0, sigma2, rng)
        beta_hats[t] = fit_min_norm(X, y)

    risks = np.sum((beta_hats - beta0) ** 2, axis=1)
    mean_beta = beta_hats.mean(axis=0)
    emp_bias2 = float(np.sum((mean_beta - beta0) ** 2))
    emp_variance = float(np.mean(np.sum((beta_hats - mean_beta) ** 2, axis=1)))
    return risks, emp_bias2, emp_variance


def run_grid(n: int, gammas, r2: float, sigma2: float, n_trials: int,
              seed: int, min_gap: int = 2):
    """Sweep gamma = p/n, skipping configs within `min_gap` of the p == n
    interpolation threshold where the exact theory formula is singular.
    """
    results = []
    for i, gamma in enumerate(gammas):
        p = max(1, round(gamma * n))
        if abs(p - n) < min_gap:
            continue
        risks, emp_bias2, emp_variance = run_config(
            n, p, r2, sigma2, n_trials, seed=seed + i
        )
        emp_mean = float(risks.mean())
        emp_stderr = float(risks.std(ddof=1) / np.sqrt(len(risks)))
        theory = exact_risk(n, p, r2, sigma2)
        theory_bias2, theory_variance = bias_variance_exact(n, p, r2, sigma2)
        results.append({
            "n": n,
            "p": p,
            "gamma": p / n,
            "empirical_risk": emp_mean,
            "empirical_stderr": emp_stderr,
            "theoretical_risk": theory,
            "relative_error": abs(emp_mean - theory) / theory,
            "empirical_bias2": emp_bias2,
            "theoretical_bias2": theory_bias2,
            "empirical_variance": emp_variance,
            "theoretical_variance": theory_variance,
        })
    return results


def run_ridge_sweep(n: int, gammas, lambdas, r2: float, sigma2: float,
                     n_trials: int, seed: int):
    """Descriptive (non-theory-matched) sweep: does ridge regularization
    suppress the interpolation peak? Uses `fit_ridge` at each (gamma, lambda)
    pair including exactly at gamma = 1, where the ridgeless risk diverges.
    """
    results = []
    for gi, gamma in enumerate(gammas):
        p = max(1, round(gamma * n))
        beta0 = generate_beta0(p, r2)
        for li, lam in enumerate(lambdas):
            rng = np.random.default_rng(seed + gi * 1000 + li)
            risks = np.empty(n_trials)
            for t in range(n_trials):
                X, y = sample_dataset(n, p, beta0, sigma2, rng)
                beta_hat = fit_ridge(X, y, lam)
                risks[t] = np.sum((beta_hat - beta0) ** 2)
            results.append({
                "n": n,
                "p": p,
                "gamma": p / n,
                "lambda": lam,
                "empirical_risk": float(risks.mean()),
                "empirical_stderr": float(risks.std(ddof=1) / np.sqrt(n_trials)),
            })
    return results
