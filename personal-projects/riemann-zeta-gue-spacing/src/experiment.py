"""Orchestrates the two sweeps this project runs:

- Height sweep: fixed window size N, four non-overlapping windows at
  increasing zero index n (i.e. increasing height T on the critical
  line), checking whether the fit to GUE statistics holds consistently
  across ~4 orders of magnitude in T.
- N-scaling sweep: nested prefixes of a single window (same height,
  growing N), checking whether the residual gap between the empirical
  distribution and the GUE prediction shrinks at the ~1/sqrt(N) rate
  expected from finite-sample noise alone.
"""

import numpy as np
from scipy.stats import linregress

from src.gue_theory import gue_surmise_cdf, poisson_cdf, montgomery_pair_correlation
from src.statistics_tests import (
    ks_against_cdf,
    pair_correlation_histogram,
    pair_correlation_l2_error,
    repulsion_fraction,
)
from src.unfolding import spacings
from src.zeta_zeros import zero_heights
from src.unfolding import unfold_heights


def _metrics_from_unfolded(x, u_max=3.0, n_bins=20, repulsion_threshold=0.2):
    s = spacings(x)
    ks_gue, p_gue = ks_against_cdf(s, gue_surmise_cdf)
    ks_poisson, p_poisson = ks_against_cdf(s, poisson_cdf)
    bin_centers, r2_hat = pair_correlation_histogram(x, u_max=u_max, n_bins=n_bins)
    pc_err_montgomery = pair_correlation_l2_error(bin_centers, r2_hat, montgomery_pair_correlation)
    pc_err_flat = pair_correlation_l2_error(bin_centers, r2_hat, lambda u: np.ones_like(u))
    return {
        "n_spacings": len(s),
        "mean_spacing": float(s.mean()),
        "std_spacing": float(s.std()),
        "ks_gue": ks_gue,
        "p_gue": p_gue,
        "ks_poisson": ks_poisson,
        "p_poisson": p_poisson,
        "pc_err_montgomery": pc_err_montgomery,
        "pc_err_flat": pc_err_flat,
        "repulsion_fraction": repulsion_fraction(s, threshold=repulsion_threshold),
        "bin_centers": bin_centers,
        "r2_hat": r2_hat,
        "spacings": s,
    }


def analyze_window(n_start, count, u_max=3.0, n_bins=20, dps=25, label=None):
    """Fetches `count` consecutive zeta zeros starting at index `n_start`,
    unfolds them, and computes the full metric set."""
    heights = zero_heights(n_start, count, dps=dps)
    x = unfold_heights(heights)
    metrics = _metrics_from_unfolded(x, u_max=u_max, n_bins=n_bins)
    metrics.update(
        n_start=n_start,
        count=count,
        t_min=heights[0],
        t_max=heights[-1],
        label=label or f"n={n_start}",
        unfolded=x,
    )
    return metrics


def nested_subsample_metrics(window_result, n_values, u_max=3.0, n_bins=20):
    """Recomputes metrics on growing prefixes of an already-unfolded
    window, to study how the fit to GUE improves with sample size N at
    fixed height. n_values must not exceed window_result['count'].
    """
    x_full = window_result["unfolded"]
    out = []
    for n_value in n_values:
        if n_value + 1 > len(x_full):
            raise ValueError(
                f"requested N={n_value} needs {n_value + 1} zeros, "
                f"window only has {len(x_full)}"
            )
        x_sub = x_full[: n_value + 1]
        metrics = _metrics_from_unfolded(x_sub, u_max=u_max, n_bins=n_bins)
        metrics.update(n_start=window_result["n_start"], count=n_value, label=f"N={n_value}")
        out.append(metrics)
    return out


def loglog_fit(x_values, y_values):
    """Slope (and stderr, R^2) of log(y) vs log(x) via OLS -- used to
    check the ~N^-1/2 finite-sample-noise scaling prediction."""
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
