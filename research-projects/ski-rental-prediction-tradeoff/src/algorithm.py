"""The lambda-parameterized buy-threshold algorithm family for ski-rental
with predictions.

Problem: you need skis for an unknown number of days x (revealed only in
hindsight). Each day costs 1 to rent; on any day you may instead buy for a
one-time cost b (skis free forever after). Offline optimum is
OPT(x, b) = min(x, b).

Given a prediction y of x and a trust parameter lambda in [0, 1], the
algorithm commits, in advance, to a buy-day threshold tau(y, lambda, b) and
then rents through day tau - 1, buying on day tau if it is still skiing.

    if y >= b:
        tau = round(1 + (1 - lambda) * (b - 1))
    else:
        tau = round((y + 1) + (1 - lambda) * (b - y - 1))
    tau = clip(tau, 1, b)

At lambda = 1 (full trust): if y >= b, tau = 1 (buy immediately -- the
predictor says "long trip"); if y < b, tau = y + 1 (defer buying to one day
past the predicted end) -- i.e. follow the prediction exactly. At lambda = 0
(no trust): tau = b regardless of y, which is exactly the classical
worst-case-optimal deterministic strategy (rent through day b-1, buy on day
b), the well-known tight 2-competitive algorithm.
"""

from __future__ import annotations

import numpy as np


def tau(y: float, lam: float, b: int) -> int:
    """Scalar buy-day threshold. y and b are in "days" (b >= 1), lam in [0,1].

    y may be any non-negative real/int prediction of the true trip length x.
    Returns an integer in [1, b].
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if y < 0:
        raise ValueError(f"y must be >= 0, got {y}")

    if y >= b:
        t = 1 + (1 - lam) * (b - 1)
    else:
        t = (y + 1) + (1 - lam) * (b - y - 1)

    t = round(t)
    t = min(max(t, 1), b)
    return int(t)


def tau_vec(y_arr: np.ndarray, lam: float, b: int) -> np.ndarray:
    """Vectorized buy-day threshold over an array of predictions y_arr.

    Numerically identical to calling tau() elementwise (same round() /
    np.round() half-to-even convention), but avoids a Python-level loop so
    it is fast enough for brute-force search and Monte Carlo estimation at
    b in the thousands and tens of thousands of samples.
    """
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    y_arr = np.asarray(y_arr, dtype=float)
    if np.any(y_arr < 0):
        raise ValueError("all entries of y_arr must be >= 0")

    branch_high = 1 + (1 - lam) * (b - 1)
    branch_low = (y_arr + 1) + (1 - lam) * (b - y_arr - 1)
    t = np.where(y_arr >= b, branch_high, branch_low)
    t = np.round(t)
    t = np.clip(t, 1, b)
    return t.astype(np.int64)


def opt(x, b: int):
    """Offline optimum: min(x, b). Works on scalars or numpy arrays."""
    return np.minimum(x, b)


def cost(x, tau_val, b: int):
    """Realized cost of the buy-at-tau strategy against true outcome x.

    cost(x) = x                if x < tau   (trip ended before the buy day)
            = (tau - 1) + b    if x >= tau   (rented tau-1 days, then bought)

    Works elementwise on numpy arrays for x and/or tau_val.
    """
    x_arr = np.asarray(x)
    tau_arr = np.asarray(tau_val)
    result = np.where(x_arr < tau_arr, x_arr, (tau_arr - 1) + b)
    if result.ndim == 0:
        return result.item()
    return result


def competitive_ratio(x, tau_val, b: int):
    """cost(x, tau, b) / OPT(x, b).

    Undefined at x = 0 (both cost and OPT are 0 there); by convention we
    return 1.0 (no rental days elapsed, no waste possible) rather than NaN,
    which keeps aggregation code simple. Every experiment in this project
    only ever evaluates x >= 1, so this convention is never load-bearing --
    it exists purely so unit tests can exercise the x = 0 edge case without
    special-casing it at every call site.
    """
    x_arr = np.asarray(x, dtype=float)
    o = np.minimum(x_arr, b).astype(float)
    c = np.asarray(cost(x_arr, tau_val, b), dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(o == 0, 1.0, c / np.where(o == 0, 1.0, o))

    if ratio.ndim == 0:
        return float(ratio)
    return ratio
