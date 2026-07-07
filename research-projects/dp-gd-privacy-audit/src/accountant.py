"""Renyi-DP accountant for the composed Gaussian mechanism used by DP-GD.

Mechanism per step: sum of per-example gradients (each clipped to L2 norm
<= C), sensitivity C under add/remove-one-record neighboring relation
(removing the canary changes the sum by exactly one vector of norm C),
Gaussian noise N(0, (sigma*C)^2 I) added to the sum.

This is a scaled Gaussian mechanism: noise-std / sensitivity = sigma. The
per-step Renyi-DP at order alpha for the Gaussian mechanism depends only on
this ratio (Mironov 2017, Prop. 7):

    rdp_alpha(1) = alpha / (2 * sigma**2)

RDP composes additively across T adaptive steps as long as the per-step
sensitivity bound holds unconditionally (true here: clipping enforces the
sensitivity bound regardless of the -- possibly data- and noise-dependent --
current parameters), so:

    rdp_alpha(T) = T * alpha / (2 * sigma**2)

Converting RDP to (epsilon, delta)-DP (Mironov 2017, Prop. 3):

    epsilon(delta) = min_{alpha > 1} [ rdp_alpha(T) + ln(1/delta) / (alpha - 1) ]
"""
from __future__ import annotations

import math

import numpy as np

_ALPHA_GRID = np.concatenate([np.linspace(1.01, 10, 500), np.linspace(10, 1000, 500)])


def rdp_alpha(alpha: float, sigma: float, T: int = 1) -> float:
    """Renyi-DP at order alpha after T composed steps of the Gaussian mechanism.

    rdp_alpha(T) = T * alpha / (2 * sigma**2)

    alpha must be > 1 (strictly) for RDP to be defined at that order; sigma
    must be > 0. T is the number of composed steps (T >= 1).
    """
    if sigma <= 0:
        return math.inf
    return T * alpha / (2.0 * sigma ** 2)


def epsilon_from_rdp(sigma: float, T: int, delta: float, alpha_grid: np.ndarray | None = None
                      ) -> tuple[float, float]:
    """Convert composed RDP to (epsilon, delta)-DP via a numeric grid search.

    epsilon(delta) = min over alpha in alpha_grid of
        [ rdp_alpha(T) + ln(1/delta) / (alpha - 1) ]

    Returns (epsilon, argmin_alpha). Degenerate inputs (delta <= 0, delta >= 1,
    sigma <= 0, T < 1) return (inf, nan) rather than raising.
    """
    if delta <= 0 or delta >= 1 or sigma <= 0 or T < 1:
        return math.inf, math.nan

    grid = _ALPHA_GRID if alpha_grid is None else alpha_grid
    log_inv_delta = math.log(1.0 / delta)
    rdp_vals = T * grid / (2.0 * sigma ** 2)
    candidate_eps = rdp_vals + log_inv_delta / (grid - 1.0)
    idx = int(np.argmin(candidate_eps))
    return float(candidate_eps[idx]), float(grid[idx])


def classical_gaussian_epsilon(sigma: float, delta: float) -> float:
    """Classical Dwork-Roth Gaussian-mechanism (epsilon, delta) bound, T=1 only.

    epsilon = sqrt(2 * ln(1.25 / delta)) / sigma

    This is a cross-check reference only -- not used elsewhere in the audit
    pipeline (the RDP accountant above is the one actually used).
    """
    if sigma <= 0 or delta <= 0 or delta >= 1:
        return math.inf
    return math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
