"""Statistics helpers: standard error / bootstrap CIs, critical-alpha
localization via interpolation, finite-size scaling extrapolation, and
linear-vs-power-law model comparison for alpha_c(rho).

Named ``stats_utils`` (not ``stats``) to avoid shadowing ``scipy.stats``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit


def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
    """Sample mean and standard error of the mean."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, sem


def bootstrap_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = (1 - ci) / 2 * 100
    hi = (1 - (1 - ci) / 2) * 100
    return float(np.percentile(boot_means, lo)), float(np.percentile(boot_means, hi))


def find_critical_alpha(
    alphas: np.ndarray,
    mean_overlaps: np.ndarray,
    threshold: float = 0.95,
) -> float | None:
    """Locate alpha_c as the alpha at which the mean-overlap curve crosses
    `threshold` on its way down, via linear interpolation between the last
    grid point at/above threshold and the first subsequent point below it.

    Assumes `alphas` is sorted ascending. Returns None if the curve never
    crosses the threshold within the tested range (e.g. always above, or
    always below).
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    mean_overlaps = np.asarray(mean_overlaps, dtype=np.float64)
    order = np.argsort(alphas)
    alphas = alphas[order]
    mean_overlaps = mean_overlaps[order]

    above = mean_overlaps >= threshold
    if not above.any():
        return None  # never reaches threshold -- no retrieval phase observed
    if above.all():
        return None  # never fails -- transition beyond tested range

    # Last index where curve is still at/above threshold, scanning from low alpha.
    last_above_idx = np.max(np.where(above)[0])
    if last_above_idx == len(alphas) - 1:
        return None  # above-threshold at the largest alpha tested

    a0, a1 = alphas[last_above_idx], alphas[last_above_idx + 1]
    m0, m1 = mean_overlaps[last_above_idx], mean_overlaps[last_above_idx + 1]
    if m1 == m0:
        return float(a0)
    frac = (threshold - m0) / (m1 - m0)
    return float(a0 + frac * (a1 - a0))


@dataclass
class LinearExtrapolation:
    slope: float
    intercept: float  # alpha_c(infinity) estimate
    intercept_stderr: float
    r_value: float


def finite_size_extrapolate(ns: np.ndarray, alpha_cs: np.ndarray) -> LinearExtrapolation:
    """Linear fit of alpha_c(N) against 1/N; the intercept is the
    extrapolated alpha_c(infinity)."""
    ns = np.asarray(ns, dtype=np.float64)
    alpha_cs = np.asarray(alpha_cs, dtype=np.float64)
    inv_n = 1.0 / ns
    result = sp_stats.linregress(inv_n, alpha_cs)
    return LinearExtrapolation(
        slope=float(result.slope),
        intercept=float(result.intercept),
        intercept_stderr=float(result.intercept_stderr),
        r_value=float(result.rvalue),
    )


@dataclass
class CapacityFitComparison:
    alpha0: float
    linear_pred: np.ndarray
    linear_rmse: float
    linear_r2: float
    power_k: float
    power_pred: np.ndarray
    power_rmse: float
    power_r2: float
    better_model: str
    spearman_rho: float
    spearman_p: float


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_capacity_vs_rho(rhos: np.ndarray, alpha_cs: np.ndarray) -> CapacityFitComparison:
    """Compare the linear ansatz H1 (alpha_c(rho) = alpha_c(0) * (1 - rho))
    against a power-law alternative (alpha_c(rho) = alpha_c(0) * (1 - rho)^k,
    k fit by least squares), both anchored at the empirical alpha_c(0).

    Also reports a Spearman rank-correlation test for monotonic decrease of
    alpha_c with rho.
    """
    rhos = np.asarray(rhos, dtype=np.float64)
    alpha_cs = np.asarray(alpha_cs, dtype=np.float64)

    zero_idx = np.argmin(np.abs(rhos))
    alpha0 = float(alpha_cs[zero_idx])

    linear_pred = alpha0 * (1 - rhos)

    def powerlaw(rho, k):
        return alpha0 * np.power(1 - rho, k)

    try:
        popt, _ = curve_fit(powerlaw, rhos, alpha_cs, p0=[1.0], maxfev=10000)
        k_fit = float(popt[0])
    except RuntimeError:
        k_fit = 1.0
    power_pred = powerlaw(rhos, k_fit)

    linear_rmse = _rmse(alpha_cs, linear_pred)
    power_rmse = _rmse(alpha_cs, power_pred)
    linear_r2 = _r2(alpha_cs, linear_pred)
    power_r2 = _r2(alpha_cs, power_pred)

    better = "linear (H1)" if linear_rmse <= power_rmse else "power-law"

    spearman_res = sp_stats.spearmanr(rhos, alpha_cs)

    return CapacityFitComparison(
        alpha0=alpha0,
        linear_pred=linear_pred,
        linear_rmse=linear_rmse,
        linear_r2=linear_r2,
        power_k=k_fit,
        power_pred=power_pred,
        power_rmse=power_rmse,
        power_r2=power_r2,
        better_model=better,
        spearman_rho=float(spearman_res.correlation),
        spearman_p=float(spearman_res.pvalue),
    )
