"""Recovery metrics and phase-transition curve fitting."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import linregress


def skeleton_shd(estimated: np.ndarray, true: np.ndarray) -> int:
    """Structural Hamming distance between two undirected skeletons: the
    number of node pairs whose adjacency (present/absent) disagrees."""
    p = estimated.shape[0]
    iu = np.triu_indices(p, k=1)
    return int(np.sum(estimated[iu] != true[iu]))


def skeleton_precision_recall(estimated: np.ndarray, true: np.ndarray):
    p = estimated.shape[0]
    iu = np.triu_indices(p, k=1)
    est_edges = estimated[iu]
    true_edges = true[iu]
    tp = int(np.sum(est_edges & true_edges))
    fp = int(np.sum(est_edges & ~true_edges))
    fn = int(np.sum(~est_edges & true_edges))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def exact_recovery(estimated: np.ndarray, true: np.ndarray) -> bool:
    return skeleton_shd(estimated, true) == 0


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fit_logistic_threshold(xs: np.ndarray, probs: np.ndarray):
    """Fit a logistic curve P(x) = 1 / (1 + exp(-k (x - x0))) to recovery
    probabilities via least squares on the log-odds of clipped probs, and
    return (x0, k): the estimated 50%-recovery sample size and the
    steepness of the transition.

    Falls back to linear interpolation for x0 if the logistic fit is
    numerically degenerate (e.g., all probs on one side of 0.5).
    """
    xs = np.asarray(xs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    p_clipped = np.clip(probs, 0.02, 0.98)
    log_odds = np.log(p_clipped / (1 - p_clipped))

    if np.all(probs < 0.5) or np.all(probs > 0.5):
        # No crossing observed in range; report the boundary as a censored
        # estimate rather than extrapolating with a fragile fit.
        x0 = xs[np.argmin(np.abs(probs - 0.5))]
        return float(x0), float("nan")

    slope, intercept, *_ = linregress(xs, log_odds)
    if slope <= 0:
        x0 = xs[np.argmin(np.abs(probs - 0.5))]
        return float(x0), float("nan")
    x0 = -intercept / slope
    return float(x0), float(slope)


def n50_via_interpolation(xs: np.ndarray, probs: np.ndarray) -> float:
    """Sample size at which recovery probability crosses 0.5, found by
    linear interpolation on the empirical curve (robust, assumption-free
    complement to the logistic fit)."""
    xs = np.asarray(xs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(xs)
    xs, probs = xs[order], probs[order]
    if probs[0] >= 0.5:
        return float(xs[0])
    if probs[-1] < 0.5:
        return float(xs[-1])
    for k in range(len(xs) - 1):
        if probs[k] < 0.5 <= probs[k + 1]:
            if probs[k + 1] == probs[k]:
                return float(xs[k])
            frac = (0.5 - probs[k]) / (probs[k + 1] - probs[k])
            return float(xs[k] + frac * (xs[k + 1] - xs[k]))
    return float(xs[-1])


def loglog_slope(x_values, y_values):
    """Slope (and stderr, R^2) of log(y) vs log(x) via OLS."""
    x = np.log(np.asarray(x_values, dtype=float))
    y = np.log(np.asarray(y_values, dtype=float))
    res = linregress(x, y)
    return {
        "slope": res.slope,
        "stderr": res.stderr,
        "intercept": res.intercept,
        "r_squared": res.rvalue ** 2,
        "p_value": res.pvalue,
    }
