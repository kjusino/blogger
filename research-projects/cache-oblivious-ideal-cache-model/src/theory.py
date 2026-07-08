"""
Theoretical predictions and log-log regression fitting for the ideal-cache
matmul bounds (Hong & Kung 1981; Frigo, Leiserson, Prokop & Ramachandran
1999).

Predicted asymptotic miss counts (n x n matmul):

  naive (fixed i,j,k loop order, row-major storage) has TWO regimes,
  governed by whether the cache can hold the L = n cache lines needed to
  retain one full row-block group across the middle loop:
    - "capacity-sufficient" (M >= n * B): Theta(n^3 / B) -- it gets
      automatic *spatial* reuse (B columns sharing each cache line) even
      with no explicit blocking, but no benefit from cache *capacity*
      beyond that.
    - "thrashing" (M < n * B): Theta(n^3) -- every access misses.
    This is a step function in M, not a power law, so naive's M-scaling
    is tested separately (see run_sweep_naive_capacity_cliff) rather than
    fit alongside blocked/oblivious's M sweep.

  blocked (tile ~ sqrt(M/3)) and oblivious, under the tall-cache
  assumption (M = Omega(B^2)):
    Theta(n^3 / (B * sqrt(M)))
    -- both spatial (B) *and* temporal/capacity (sqrt(M)) reuse.

We never test the Theta() constant exactly (the theorem doesn't pin it
down); instead we test the *scaling exponents* by ordinary-least-squares
regression of log(misses) against log(the varied parameter), holding the
others fixed (and, for naive, staying within the capacity-sufficient
regime so a single power law applies). A bound of Theta(x^a) predicts a
fitted slope of a.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class FitResult:
    slope: float
    intercept: float
    stderr: float
    r_squared: float
    n_points: int

    def within(self, predicted_slope, tolerance):
        return abs(self.slope - predicted_slope) <= tolerance


def fit_power_law(x_values, y_values):
    """OLS fit of log(y) = slope * log(x) + intercept. Requires len >= 3
    and all values > 0."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 3:
        raise ValueError("need at least 3 points to fit a slope with a meaningful r^2")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit requires strictly positive x and y (misses must be > 0)")
    log_x = np.log(x)
    log_y = np.log(y)
    result = stats.linregress(log_x, log_y)
    return FitResult(
        slope=float(result.slope),
        intercept=float(result.intercept),
        stderr=float(result.stderr),
        r_squared=float(result.rvalue ** 2),
        n_points=len(x),
    )


# Predicted asymptotic exponents, by (experiment axis, algorithm).
# naive's M-scaling is a step function, not a power law -- it is
# deliberately absent here; see run_sweep_naive_capacity_cliff.
PREDICTED_EXPONENT = {
    ("n", "naive"): 3.0,
    ("n", "blocked"): 3.0,
    ("n", "oblivious"): 3.0,
    ("B", "naive"): -1.0,
    ("B", "blocked"): -1.0,
    ("B", "oblivious"): -1.0,
    ("M", "blocked"): -0.5,
    ("M", "oblivious"): -0.5,
}


def leading_constant(n, B, M, misses, algorithm):
    """Recover the implied leading constant c in misses ~ c * f(n, B, M)
    for the given algorithm's predicted functional form (assuming naive
    is in its capacity-sufficient regime, M >= n * B), so constants can
    be compared across algorithms/configs (not just exponents)."""
    if algorithm == "naive":
        f = (n ** 3) / B
    elif algorithm in ("blocked", "oblivious"):
        f = (n ** 3) / (B * (M ** 0.5))
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    return misses / f
