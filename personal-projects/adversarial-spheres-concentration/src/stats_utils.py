"""Small statistics helpers: log-log power-law fitting with confidence intervals."""

import numpy as np
from scipy import stats


def fit_power_law(x, y):
    """Fit y = c * x^a via OLS on log(y) = log(c) + a*log(x).

    Returns dict with exponent, exponent_se, exponent_ci95 (tuple), intercept c,
    r_squared, and n. Drops any non-positive/non-finite (x, y) pairs first.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    n = len(lx)
    if n < 3:
        return {"exponent": np.nan, "exponent_se": np.nan, "exponent_ci95": (np.nan, np.nan),
                "c": np.nan, "r_squared": np.nan, "n": n}
    res = stats.linregress(lx, ly)
    tval = stats.t.ppf(0.975, df=n - 2)
    ci = (res.slope - tval * res.stderr, res.slope + tval * res.stderr)
    return {
        "exponent": float(res.slope),
        "exponent_se": float(res.stderr),
        "exponent_ci95": (float(ci[0]), float(ci[1])),
        "c": float(np.exp(res.intercept)),
        "r_squared": float(res.rvalue ** 2),
        "n": n,
    }
