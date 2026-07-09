"""Fit the empirical asymptotic geometric decay rate of a Sinkhorn residual
history: residual_k ~ A * rate^k for large k, so log(residual_k) ~ log(A) +
k*log(rate). We fit a line to the *tail* of the (iteration, log residual)
curve, since early iterations are a pre-asymptotic transient the theory does
not describe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RateFit:
    rate: float          # exp(slope); NaN if unfittable
    r_squared: float      # goodness of the tail log-linear fit; NaN if unfittable
    n_points: int          # number of tail points used


def fit_asymptotic_rate(
    residual_history: np.ndarray,
    tail_fraction: float = 0.4,
    min_points: int = 8,
    max_points: int = 500,
) -> RateFit:
    hist = np.asarray(residual_history, dtype=float)
    positive_idx = np.flatnonzero(hist > 0)

    if len(positive_idx) < min_points:
        return RateFit(rate=float("nan"), r_squared=float("nan"), n_points=int(len(positive_idx)))

    n_tail = max(min_points, int(round(len(positive_idx) * tail_fraction)))
    n_tail = min(n_tail, max_points, len(positive_idx))
    tail_idx = positive_idx[-n_tail:]

    x = tail_idx.astype(float)
    y = np.log(hist[tail_idx])

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    rate = float(np.exp(slope))
    return RateFit(rate=rate, r_squared=float(r_squared), n_points=int(n_tail))
