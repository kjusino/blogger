"""Monte Carlo experiment: finite-length BEC decoding vs. DE threshold.

For a fixed (dv, dc)-regular ensemble, this module measures the empirical
block-error rate (BLER) of the from-scratch peeling decoder as a function
of erasure probability epsilon, for a range of blocklengths n. It then
fits a logistic curve to each blocklength's BLER-vs-epsilon data to extract:

  - eps50(n): the erasure rate at which BLER crosses 50%
  - width(n): the eps_90 - eps_10 "waterfall" transition width

and fits power laws to how eps*-eps50(n) and width(n) shrink with n, to
compare against the finite-length scaling theory of Amraoui, Montanari,
Richardson & Urbanke ("Finite-Length Scaling for Iteratively Decoded LDPC
Ensembles", IEEE Trans. Inform. Theory, 2009), which predicts a threshold-
distance shift ~ n^(-2/3) and a transition width ~ n^(-1/2).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .decoder import peel_decode, sample_erasures
from .tanner import build_regular_tanner_graph


def _logistic(eps: np.ndarray, eps50: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (eps - eps50)))


def _logit_regression_guess(eps_arr: np.ndarray, bler_arr: np.ndarray) -> tuple[float, float]:
    """Initial (eps50, k) guess via linear regression on the logit scale.

    log(p/(1-p)) = k*eps - k*eps50 is linear in eps, so an ordinary
    least-squares fit on clipped, logit-transformed BLER gives a much
    better-conditioned starting point for the nonlinear logistic fit than a
    fixed guess (a fixed large k, e.g. 200, can trap curve_fit in a
    spuriously steep local optimum when the true transition is gradual --
    this showed up empirically on small-n, noisy Monte Carlo curves).
    """
    clipped = np.clip(bler_arr, 0.02, 0.98)
    logit = np.log(clipped / (1 - clipped))
    k_guess, intercept = np.polyfit(eps_arr, logit, 1)
    if k_guess <= 0 or not math.isfinite(k_guess):
        span = eps_arr[-1] - eps_arr[0]
        k_guess = 4.394 / max(span, 1e-6)
        eps50_guess = float(np.interp(0.5, np.clip(bler_arr, 0, 1), eps_arr))
    else:
        eps50_guess = -intercept / k_guess
    return float(eps50_guess), float(k_guess)


def fit_logistic(epsilons: list[float], blers: list[float]) -> tuple[float, float]:
    """Fit BLER(eps) ~ logistic(eps; eps50, k); returns (eps50, k).

    Falls back to the logit-regression guess itself (see
    `_logit_regression_guess`) if the nonlinear refinement fails to
    converge (can happen with very noisy or degenerate data, e.g. all-0 or
    all-1 BLER).
    """
    eps_arr = np.asarray(epsilons, dtype=float)
    bler_arr = np.asarray(blers, dtype=float)
    order = np.argsort(eps_arr)
    eps_arr, bler_arr = eps_arr[order], bler_arr[order]

    eps50_guess, k_guess = _logit_regression_guess(eps_arr, bler_arr)
    try:
        popt, _ = curve_fit(
            _logistic,
            eps_arr,
            bler_arr,
            p0=[eps50_guess, k_guess],
            maxfev=20000,
            bounds=([0.0, 1.0], [1.0, 1e6]),
        )
        eps50, k = float(popt[0]), float(popt[1])
        if not math.isfinite(eps50) or not math.isfinite(k) or k <= 0:
            raise RuntimeError("non-finite or non-positive fit params")
        return eps50, k
    except Exception:
        return eps50_guess, k_guess


def waterfall_width(k: float) -> float:
    """eps_90 - eps_10 for a logistic curve with steepness k."""
    return 2.0 * math.log(9.0) / k


@dataclass
class BlerCurve:
    n: int
    epsilons: list[float] = field(default_factory=list)
    blers: list[float] = field(default_factory=list)
    trials: list[int] = field(default_factory=list)
    eps50: float = 0.0
    steepness: float = 0.0
    width_90_10: float = 0.0


def measure_bler(
    n: int,
    dv: int,
    dc: int,
    epsilon: float,
    n_graph_instances: int,
    trials_per_graph: int,
    rng: random.Random,
) -> float:
    """Empirical BLER at a single epsilon, averaged over random code and
    channel realizations (standard practice: average over both the code
    ensemble and the channel to estimate ensemble-average performance)."""
    failures = 0
    total = 0
    for _ in range(n_graph_instances):
        graph = build_regular_tanner_graph(n, dv, dc, rng)
        for _ in range(trials_per_graph):
            erased = sample_erasures(n, epsilon, rng)
            ok, _ = peel_decode(graph, erased)
            failures += not ok
            total += 1
    return failures / total


def measure_bler_curve(
    n: int,
    dv: int,
    dc: int,
    eps_star: float,
    rng: random.Random,
    n_graph_instances: int = 3,
    coarse_trials: int = 40,
    fine_trials: int = 150,
    coarse_half_width: float = 0.10,
    n_coarse: int = 7,
    n_fine: int = 9,
) -> BlerCurve:
    """Two-stage adaptive sweep: a coarse pass locates the transition
    region near eps_star, then a fine pass (narrower window, more trials)
    resolves it precisely. The fine window narrows automatically with the
    coarse fit's steepness estimate, since the waterfall region shrinks as
    n grows.
    """
    coarse_eps = np.clip(
        np.linspace(eps_star - coarse_half_width, eps_star + coarse_half_width, n_coarse),
        1e-4,
        1 - 1e-4,
    ).tolist()
    curve = BlerCurve(n=n)
    for eps in coarse_eps:
        bler = measure_bler(n, dv, dc, eps, n_graph_instances, coarse_trials, rng)
        curve.epsilons.append(eps)
        curve.blers.append(bler)
        curve.trials.append(n_graph_instances * coarse_trials)

    rough_eps50, rough_k = fit_logistic(curve.epsilons, curve.blers)
    fine_half_width = min(coarse_half_width, max(4.0 / rough_k, 0.005))
    fine_eps = np.clip(
        np.linspace(rough_eps50 - fine_half_width, rough_eps50 + fine_half_width, n_fine),
        1e-4,
        1 - 1e-4,
    ).tolist()
    for eps in fine_eps:
        bler = measure_bler(n, dv, dc, eps, n_graph_instances, fine_trials, rng)
        curve.epsilons.append(eps)
        curve.blers.append(bler)
        curve.trials.append(n_graph_instances * fine_trials)

    curve.eps50, curve.steepness = fit_logistic(curve.epsilons, curve.blers)
    curve.width_90_10 = waterfall_width(curve.steepness)
    return curve


def fit_power_law(ns: list[int], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of log(y) = log(C) + p*log(n); returns (C, p).

    Only uses entries with y > 0 (a zero gap/width can't be log-fit and
    typically only happens from Monte Carlo noise at the largest n).
    """
    ns_arr = np.asarray(ns, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    mask = ys_arr > 0
    if mask.sum() < 2:
        raise ValueError("need at least 2 positive points to fit a power law")
    log_n = np.log(ns_arr[mask])
    log_y = np.log(ys_arr[mask])
    p, log_c = np.polyfit(log_n, log_y, 1)
    return float(math.exp(log_c)), float(p)
