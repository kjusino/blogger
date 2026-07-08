"""Theoretical predictions for the surface-code threshold theorem, and
estimators that pull the same quantities out of simulated data.

Threshold theorem (Dennis, Kitaev, Landahl & Preskill 2002; Fowler et al.
2012): under a local, sufficiently unbiased noise model, there exists a
threshold error rate p_th such that for p < p_th the logical error rate
P_L(d, p) is *suppressed* as the code distance d grows, and for p > p_th
it is *amplified* as d grows. Plotted as P_L against p for several d,
this means the curves cross at (approximately) a single point p_th.

Below threshold, P_L is dominated by the lowest-weight logical operator,
which requires on the order of ceil(d/2) independent physical faults to
trigger (a decoder correctly resolves any error of weight < d/2). So to
leading order P_L(d, p) ~ A(p/p_th)^floor((d+1)/2), i.e. log P_L is
linear in log p with slope floor((d+1)/2).
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


def predicted_subthreshold_exponent(distance: int) -> int:
    """Leading-order power of p in the sub-threshold logical error rate
    for a distance-`distance` code: floor((d+1)/2)."""
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    return (distance + 1) // 2


@dataclass(frozen=True)
class ExponentFit:
    distance: int
    slope: float
    slope_stderr: float
    intercept: float
    r_value: float
    predicted_slope: int


def fit_subthreshold_exponent(
    p_values: np.ndarray, logical_error_rates: np.ndarray, distance: int
) -> ExponentFit:
    """Fit log(P_L) = slope * log(p) + intercept over the given (sub-
    threshold) points and compare the fitted slope to theory."""
    p_values = np.asarray(p_values, dtype=float)
    rates = np.asarray(logical_error_rates, dtype=float)
    mask = rates > 0
    if mask.sum() < 2:
        raise ValueError("need at least two nonzero-rate points to fit a slope")
    log_p = np.log(p_values[mask])
    log_rate = np.log(rates[mask])
    result = linregress(log_p, log_rate)
    return ExponentFit(
        distance=distance,
        slope=float(result.slope),
        slope_stderr=float(result.stderr),
        intercept=float(result.intercept),
        r_value=float(result.rvalue),
        predicted_slope=predicted_subthreshold_exponent(distance),
    )


def find_pairwise_crossing(
    p_grid: np.ndarray, rates_low_d: np.ndarray, rates_high_d: np.ndarray
) -> float | None:
    """Locate the physical error rate at which two distances' logical
    error rate curves cross, by linearly interpolating log(rate) between
    the grid points that bracket a sign change of
    log(rates_low_d) - log(rates_high_d).

    Returns None if the two curves never swap ordering across the grid
    (e.g. the swept range doesn't bracket threshold).
    """
    p_grid = np.asarray(p_grid, dtype=float)
    lo = np.asarray(rates_low_d, dtype=float)
    hi = np.asarray(rates_high_d, dtype=float)
    eps = np.finfo(float).tiny
    diff = np.log(np.clip(lo, eps, None)) - np.log(np.clip(hi, eps, None))

    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return None

    # Use the first crossing (lowest p): below threshold the lower-distance
    # code has fewer failure paths, so diff < 0; above threshold diff > 0.
    i = sign_changes[0]
    x0, x1 = np.log(p_grid[i]), np.log(p_grid[i + 1])
    y0, y1 = diff[i], diff[i + 1]
    # Linear interpolation for the zero-crossing of y between (x0,y0),(x1,y1).
    t = -y0 / (y1 - y0)
    log_p_star = x0 + t * (x1 - x0)
    return float(np.exp(log_p_star))


def estimate_threshold(
    p_grid: np.ndarray, rates_by_distance: dict[int, np.ndarray]
) -> tuple[float | None, dict[tuple[int, int], float | None]]:
    """Estimate the pseudo-threshold as the mean of all pairwise adjacent-
    distance crossing points. Returns (estimate, per-pair crossings)."""
    distances = sorted(rates_by_distance)
    crossings: dict[tuple[int, int], float | None] = {}
    for d_lo, d_hi in zip(distances, distances[1:]):
        crossings[(d_lo, d_hi)] = find_pairwise_crossing(
            p_grid, rates_by_distance[d_lo], rates_by_distance[d_hi]
        )
    found = [v for v in crossings.values() if v is not None]
    estimate = float(np.mean(found)) if found else None
    return estimate, crossings
