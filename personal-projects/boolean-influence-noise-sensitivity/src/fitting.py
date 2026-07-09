"""Scaling-law fits and the KKL isoperimetric-bound check."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PowerLawFit:
    exponent: float
    exponent_stderr: float
    intercept: float  # log(coefficient)
    r_squared: float

    def exponent_ci95(self) -> tuple[float, float]:
        margin = 1.96 * self.exponent_stderr
        return (self.exponent - margin, self.exponent + margin)


def power_law_fit(xs: np.ndarray, ys: np.ndarray) -> PowerLawFit:
    """Fit y = c * x^a by linear regression of log(y) on log(x).

    Returns the exponent `a`, its standard error, log(c), and R^2.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if np.any(xs <= 0) or np.any(ys <= 0):
        raise ValueError("power_law_fit requires strictly positive xs and ys")
    if len(xs) < 3:
        raise ValueError("power_law_fit needs at least 3 points")

    log_x, log_y = np.log(xs), np.log(ys)
    result = stats.linregress(log_x, log_y)
    return PowerLawFit(
        exponent=float(result.slope),
        exponent_stderr=float(result.stderr),
        intercept=float(result.intercept),
        r_squared=float(result.rvalue**2),
    )


def kkl_scaling_ratio(max_influence: float, variance: float, n: int) -> float:
    """max_influence * n / (variance * log2(n)).

    The Kahn-Kalai-Linial theorem guarantees max_i Inf_i(f) = Omega(Var(f) *
    log(n) / n) for every f: {-1,+1}^n -> {-1,+1}, i.e. this ratio is bounded
    away from 0 as n grows (for any fixed, unknown constant hidden in the
    Omega). We report the ratio itself rather than assert a specific
    constant, since the KKL constant is not made explicit by the original
    proof.
    """
    if n < 2:
        raise ValueError("kkl_scaling_ratio requires n >= 2")
    if variance <= 0:
        return float("inf")  # constant function: bound holds vacuously
    return (max_influence * n) / (variance * np.log2(n))
