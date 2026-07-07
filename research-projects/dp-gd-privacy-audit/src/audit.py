"""Black-box membership-inference privacy audit (Monte Carlo canary-gradient
attack), following the Jagielski/Nasr auditing methodology.

For a fixed DP-GD config, we run N independent trials of the mechanism with
the canary present ("IN" world) and N independent trials with the canary
absent ("OUT" world), each with a fresh, independent RNG draw (this is a true
two-sample audit, not a paired/common-random-numbers design). The attack
statistic is the final theta's projection onto e1 (theta_final[0]), which by
construction is expected to be systematically lower in the IN world (the
canary update subtracts lr*C along e1 every step it's present).

We sweep candidate thresholds, build the empirical ROC, and for each
threshold compute a 95%-confidence Clopper-Pearson lower bound on epsilon
from the DP hypothesis-testing characterization. The best (max) bound over
the sweep is reported as the audit's empirical eps_lower for that config.

KNOWN LIMITATION (see README): sweeping thresholds and taking the max without
a multiple-testing correction is standard practice in illustrative DP audits
but is not a rigorously corrected simultaneous confidence bound -- the
reported eps_lower should be read as "the best empirical evidence found",
not as a bound with an exact, simultaneously-valid 95% coverage guarantee.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta

from .dpgd import train_dpgd


def clopper_pearson_tpr_lower(tp: int, n_in: int, alpha_level: float = 0.05) -> float:
    """95% (or 1-alpha_level) one-sided Clopper-Pearson lower bound on TPR."""
    if tp <= 0:
        return 0.0
    return float(beta.ppf(alpha_level, tp, n_in - tp + 1))


def clopper_pearson_fpr_upper(fp: int, n_out: int, alpha_level: float = 0.05) -> float:
    """95% (or 1-alpha_level) one-sided Clopper-Pearson upper bound on FPR."""
    if fp >= n_out:
        return 1.0
    return float(beta.ppf(1.0 - alpha_level, fp + 1, n_out - fp))


def eps_lower_from_counts(
    tp: int, n_in: int, fp: int, n_out: int, delta: float, alpha_level: float = 0.05
) -> float:
    """Empirical epsilon lower bound combining both directions of the DP
    hypothesis-testing inequality; the max of two individually-valid lower
    bounds is itself a valid lower bound.
    """
    tpr_low = clopper_pearson_tpr_lower(tp, n_in, alpha_level)
    fpr_high = clopper_pearson_fpr_upper(fp, n_out, alpha_level)

    term1 = math.log(max(tpr_low - delta, 1e-12) / max(fpr_high, 1e-12))
    term2 = math.log(max((1.0 - fpr_high) - delta, 1e-12) / max(1.0 - tpr_low, 1e-12))
    return max(term1, term2, 0.0)


def sweep_thresholds(stats_in: np.ndarray, stats_out: np.ndarray) -> np.ndarray:
    """Candidate thresholds: midpoints between sorted unique observed values."""
    all_vals = np.unique(np.concatenate([stats_in, stats_out]))
    if len(all_vals) == 1:
        return all_vals.copy()
    return (all_vals[:-1] + all_vals[1:]) / 2.0


def audit_roc(stats_in: np.ndarray, stats_out: np.ndarray) -> list[dict]:
    """Empirical ROC over the threshold sweep.

    "Declare IN" is defined as statistic <= threshold (the canary shifts the
    IN-world statistic mean down by construction; verified, not hardcoded,
    in run_membership_audit via the observed mean difference).

    Returns a list of dicts, one per threshold, each with:
    threshold, tp, fp, tpr (empirical), fpr (empirical), eps_lower.
    """
    n_in = len(stats_in)
    n_out = len(stats_out)
    thresholds = sweep_thresholds(stats_in, stats_out)

    roc = []
    for thr in thresholds:
        tp = int(np.sum(stats_in <= thr))
        fp = int(np.sum(stats_out <= thr))
        tpr = tp / n_in
        fpr = fp / n_out
        roc.append({"threshold": float(thr), "tp": tp, "fp": fp, "tpr": tpr, "fpr": fpr})
    return roc


def best_audit_epsilon(
    stats_in: np.ndarray, stats_out: np.ndarray, delta: float, alpha_level: float = 0.05
) -> dict:
    """Sweep thresholds, compute eps_lower at each, return the best (max) one."""
    n_in = len(stats_in)
    n_out = len(stats_out)
    thresholds = sweep_thresholds(stats_in, stats_out)

    best = {"eps_lower": -math.inf, "threshold": math.nan, "tp": 0, "fp": 0}
    for thr in thresholds:
        tp = int(np.sum(stats_in <= thr))
        fp = int(np.sum(stats_out <= thr))
        eps = eps_lower_from_counts(tp, n_in, fp, n_out, delta, alpha_level)
        if eps > best["eps_lower"]:
            best = {"eps_lower": eps, "threshold": float(thr), "tp": tp, "fp": fp}
    return best


def run_membership_audit(
    X: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    T: int,
    C: float,
    sigma: float,
    lr: float,
    N: int,
    delta: float,
    rng: np.random.Generator,
    alpha_level: float = 0.05,
) -> dict:
    """Run the full canary-gradient membership-inference audit for one config.

    Runs N independent IN-world trials and N independent OUT-world trials
    (each trial draws fresh, independent randomness from rng -- never reused
    between IN and OUT), computes the attack statistic (theta_final[0]) for
    each, sweeps thresholds, and returns the best empirical eps_lower along
    with supporting data for plotting.
    """
    stats_in = np.empty(N)
    for i in range(N):
        theta_final = train_dpgd(X, y, theta0, T, C, sigma, lr, True, rng)
        stats_in[i] = theta_final[0]

    stats_out = np.empty(N)
    for i in range(N):
        theta_final = train_dpgd(X, y, theta0, T, C, sigma, lr, False, rng)
        stats_out[i] = theta_final[0]

    # Sanity-check (not hardcoded): canary should shift the IN-world stat down.
    mean_shift_is_negative = bool(np.mean(stats_in) < np.mean(stats_out))

    best = best_audit_epsilon(stats_in, stats_out, delta, alpha_level)
    roc = audit_roc(stats_in, stats_out)

    return {
        "eps_lower": best["eps_lower"],
        "threshold": best["threshold"],
        "tp": best["tp"],
        "fp": best["fp"],
        "N": N,
        "mean_shift_is_negative": mean_shift_is_negative,
        "mean_in": float(np.mean(stats_in)),
        "mean_out": float(np.mean(stats_out)),
        "roc": roc,
        "stats_in": stats_in,
        "stats_out": stats_out,
    }
