"""Sweep orchestration: estimate the sample size n50 at which the PC
algorithm's skeleton-recovery probability crosses 50%, for a grid of
(p, d) configurations, then fit the log-log scaling of n50 against p
(fixed d) and against d (fixed p) to compare against the theoretical
n = Theta(d^2 log p) sample-complexity bound.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .dag_generator import generate_faithful_dag
from .pc_algorithm import estimate_skeleton
from .metrics import (
    skeleton_shd,
    skeleton_precision_recall,
    exact_recovery,
    n50_via_interpolation,
    fit_logistic_threshold,
    loglog_slope,
)


def bonferroni_alpha(p: int, family_wise_alpha: float = 0.05) -> float:
    """Per-test significance level controlling the family-wise error rate
    at family_wise_alpha across all C(p, 2) pairs the algorithm can test.

    A fixed (n-independent) significance level has a constant per-test
    Type-I error rate: it does NOT shrink as n grows, so with p pairs
    tested, spurious edges survive at a rate that grows with p unless
    alpha is scaled down accordingly. This is exactly the role played by
    the significance-level sequence alpha_n in the PC-algorithm
    consistency theorem (Kalisch & Buhlmann, 2007); a Bonferroni
    correction over pairs is the standard practical way to realize it.
    """
    n_pairs = p * (p - 1) / 2
    return family_wise_alpha / n_pairs


@dataclass
class TrialResult:
    p: int
    d: int
    n: int
    exact: bool
    shd: int
    precision: float
    recall: float
    realized_degree: int


def max_cond_set_cap(p: int, d: int) -> int:
    """Cap on conditioning-set size explored per pair, mirroring the
    `m.max` parameter of practical PC-algorithm implementations (e.g.
    pcalg): searching the full range up to p-2 is both statistically
    unnecessary for a graph known to be d-sparse and, at small n where
    many spurious edges have not yet been pruned, combinatorially
    catastrophic (a near-complete graph would demand testing binomial(p,
    k) subsets per pair). d+3 is generous slack above the true max degree
    without being unbounded.
    """
    return min(p - 2, d + 3)


def min_n_for_power(p: int, alpha: float, min_margin: float, safety_factor: float = 1.5) -> int:
    """Smallest n at which even the weakest guaranteed true-edge signal
    (min_margin) is detectable with reasonable power at this alpha. Used
    as a floor for the search grid: starting far below this floor wastes
    compute in a regime with near-zero power, where almost no edges can
    be pruned and the skeleton search stalls near the complete graph."""
    z_alpha = norm.ppf(1 - alpha / 2)
    dof_needed = (z_alpha / np.arctanh(min_margin)) ** 2
    return int(safety_factor * dof_needed) + 10


def run_trial(p: int, d: int, n: int, alpha: float, min_margin: float, rng: np.random.Generator) -> TrialResult:
    sem = generate_faithful_dag(p=p, max_degree=d, rng=rng, min_margin=min_margin)
    X = sem.sample(n, rng)
    cov = np.cov(X, rowvar=False)
    result = estimate_skeleton(cov, n=n, alpha=alpha, max_cond_set=max_cond_set_cap(p, d))
    true_skeleton = sem.skeleton()
    shd = skeleton_shd(result.skeleton, true_skeleton)
    precision, recall = skeleton_precision_recall(result.skeleton, true_skeleton)
    return TrialResult(
        p=p, d=d, n=n, exact=exact_recovery(result.skeleton, true_skeleton), shd=shd,
        precision=precision, recall=recall, realized_degree=sem.realized_max_degree(),
    )


def recovery_probability(p: int, d: int, n: int, trials: int, alpha: float, min_margin: float, rng: np.random.Generator):
    results = [run_trial(p, d, n, alpha, min_margin, rng) for _ in range(trials)]
    prob = sum(r.exact for r in results) / len(results)
    mean_shd = sum(r.shd for r in results) / len(results)
    mean_precision = sum(r.precision for r in results) / len(results)
    mean_recall = sum(r.recall for r in results) / len(results)
    return {
        "p": p, "d": d, "n": n, "trials": trials,
        "recovery_prob": prob, "mean_shd": mean_shd,
        "mean_precision": mean_precision, "mean_recall": mean_recall,
    }


def locate_n50(
    p: int, d: int, alpha: float, min_margin: float, rng: np.random.Generator,
    coarse_trials: int = 10, coarse_points: int = 12,
    n_lo: int = None, n_hi: int = 200_000,
):
    """Two-phase search for n50: a coarse geometric sweep to bracket the
    50%-recovery crossing, used only to center the fine grid recorded by
    the caller."""
    n_lo = n_lo or max(4 * p, 20)
    grid = np.unique(np.round(np.geomspace(n_lo, n_hi, coarse_points)).astype(int))
    probs = []
    for n in grid:
        row = recovery_probability(p, d, int(n), coarse_trials, alpha, min_margin, rng)
        probs.append(row["recovery_prob"])
        if probs[-1] >= 0.9 and len(probs) >= 3:
            break  # already well past the transition, no need to keep climbing
    grid = grid[: len(probs)]
    return n50_via_interpolation(grid, np.array(probs)), grid, np.array(probs)


def fine_grid_around(center: float, n_lo_floor: int, points: int = 7, span: float = 6.0):
    lo = max(n_lo_floor, center / span)
    hi = center * span
    grid = np.unique(np.round(np.geomspace(lo, hi, points)).astype(int))
    return grid


def run_config(
    p: int, d: int, rng: np.random.Generator,
    family_wise_alpha: float = 0.05, min_margin: float = 0.1,
    coarse_trials: int = 10, fine_trials: int = 30, fine_points: int = 7,
):
    """Run the full two-phase estimation for one (p, d) configuration.

    Returns (n50_estimate, curve_rows) where curve_rows is a list of dicts
    ready to append to the results table (one row per n on the fine grid).
    """
    alpha = bonferroni_alpha(p, family_wise_alpha)
    # The floor only needs to keep the search out of the near-complete-graph
    # regime long enough for max_cond_set_cap to bound worst-case cost; it
    # does NOT need to be above the power threshold (min_n_for_power) --
    # that would hide the transition itself, since aggregate multi-edge
    # recovery can cross 50% well below the single-worst-edge power point.
    n_lo_floor = max(2 * p, 20)
    coarse_n50, _, _ = locate_n50(p, d, alpha, min_margin, rng, coarse_trials=coarse_trials, n_lo=n_lo_floor)
    grid = fine_grid_around(coarse_n50, n_lo_floor, points=fine_points)

    rows = []
    for n in grid:
        row = recovery_probability(p, d, int(n), fine_trials, alpha, min_margin, rng)
        row["alpha"] = alpha
        rows.append(row)

    ns = np.array([r["n"] for r in rows])
    probs = np.array([r["recovery_prob"] for r in rows])
    n50_interp = n50_via_interpolation(ns, probs)
    x0_logistic, steepness = fit_logistic_threshold(ns, probs)

    return {
        "p": p, "d": d, "alpha": alpha, "n50_interp": n50_interp,
        "n50_logistic": x0_logistic, "logistic_steepness": steepness,
        "rows": rows,
    }


def run_p_sweep(p_values, d_fixed: int, rng: np.random.Generator, **kwargs):
    configs = [run_config(p=p, d=d_fixed, rng=rng, **kwargs) for p in p_values]
    fit = loglog_slope([c["p"] for c in configs], [c["n50_interp"] for c in configs])
    return configs, fit


def run_d_sweep(d_values, p_fixed: int, rng: np.random.Generator, **kwargs):
    configs = [run_config(p=p_fixed, d=d, rng=rng, **kwargs) for d in d_values]
    fit = loglog_slope([c["d"] for c in configs], [c["n50_interp"] for c in configs])
    return configs, fit
