"""Growth-model fitting for max-load-gap vs n curves.

We fit two competing models to (n, gap) data:

  log-model:     gap(n) = A * ln(n) + B
  loglog-model:  gap(n) = A * ln(ln(n)) + B

and report whichever gives the higher R^2, plus the R^2 of both so the
caller can see how decisive the comparison is. This operationalizes
"does this family's growth look logarithmic or log-logarithmic?" as a
single number instead of an eyeballed plot.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _log_model(n, a, b):
    return a * np.log(n) + b


def _loglog_model(n, a, b):
    return a * np.log(np.log(n)) + b


@dataclass
class FitResult:
    log_params: tuple
    log_r2: float
    loglog_params: tuple
    loglog_r2: float

    @property
    def best_model(self) -> str:
        return "loglog" if self.loglog_r2 >= self.log_r2 else "log"


def _r_squared(y, y_pred):
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_growth_models(ns, gaps) -> FitResult:
    """Fit both models to (ns, gaps); ns must have at least 2 points > 1."""
    ns = np.asarray(ns, dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    if len(ns) < 3:
        raise ValueError("need at least 3 points to fit a 2-parameter model")
    if np.any(ns <= 1):
        raise ValueError("fitting requires n > 1 (log(log(n)) undefined otherwise)")

    log_params, _ = curve_fit(_log_model, ns, gaps, p0=(1.0, 0.0), maxfev=10000)
    loglog_params, _ = curve_fit(_loglog_model, ns, gaps, p0=(1.0, 0.0), maxfev=10000)

    log_r2 = _r_squared(gaps, _log_model(ns, *log_params))
    loglog_r2 = _r_squared(gaps, _loglog_model(ns, *loglog_params))

    return FitResult(
        log_params=tuple(log_params),
        log_r2=log_r2,
        loglog_params=tuple(loglog_params),
        loglog_r2=loglog_r2,
    )
