"""Model-comparison fits for the barren-plateau scaling hypothesis.

Two competing models for how Var[dC/dtheta] falls off with the number of
qubits n, at fixed depth:

  exponential (barren plateau):  log2 Var = a - b * n           (n-linear in log space)
  power law (no plateau):        log2 Var = a - b * log2(n)     (log(n)-linear in log space)

Both are fit by ordinary least squares in log space and compared by R^2.
A true barren plateau should be won cleanly by the exponential model with a
large b; its absence should favor the power-law model or at least yield a
much smaller effective b.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float


def _ols_r2(x: np.ndarray, y: np.ndarray) -> FitResult:
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return FitResult(slope=float(slope), intercept=float(intercept), r2=r2)


def fit_exponential(n_values, variances) -> FitResult:
    x = np.asarray(n_values, dtype=float)
    y = np.log2(np.asarray(variances, dtype=float))
    return _ols_r2(x, y)


def fit_power_law(n_values, variances) -> FitResult:
    x = np.log2(np.asarray(n_values, dtype=float))
    y = np.log2(np.asarray(variances, dtype=float))
    return _ols_r2(x, y)
