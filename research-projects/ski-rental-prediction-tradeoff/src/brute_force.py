"""From-scratch numeric (brute-force) search for the worst-case Robustness
and Consistency ratios, used to validate the closed forms in theory.py.

No shortcut is taken based on the theory-derived location of the worst
case: both functions scan the full stated ranges (x in 1..x_mult*b, and for
robustness also y in 0..y_mult*b) and take the max ratio actually observed.
The only thing that is "clever" is using numpy array operations instead of
a nested Python double loop, purely for speed -- the search space itself is
exactly the one specified (x from 1 to 4b, y from 0 to 2b by default; ratio
becomes constant for x >= b so 4b is generous headroom).

x = 0 is excluded from the scan: OPT(0, b) = 0 makes the ratio undefined
there (see algorithm.competitive_ratio's docstring for the 0/0 convention),
and it can never be the worst case since cost(0, ...) = 0 too.
"""

from __future__ import annotations

import numpy as np

from .algorithm import tau, tau_vec


def brute_force_robustness(
    lam: float, b: int, x_mult: int = 4, y_mult: int = 2
):
    """Max ratio over the joint grid of x in [1, x_mult*b] and
    y in [0, y_mult*b].

    Returns (best_ratio, best_x, best_y).
    """
    x_max = x_mult * b
    y_max = y_mult * b

    x_arr = np.arange(1, x_max + 1, dtype=np.int64)
    opt_arr = np.minimum(x_arr, b).astype(float)

    best_ratio = -np.inf
    best_x = None
    best_y = None

    for y in range(0, y_max + 1):
        t = tau(y, lam, b)
        cost_arr = np.where(x_arr < t, x_arr, (t - 1) + b).astype(float)
        ratio_arr = cost_arr / opt_arr
        idx = int(np.argmax(ratio_arr))
        if ratio_arr[idx] > best_ratio:
            best_ratio = float(ratio_arr[idx])
            best_x = int(x_arr[idx])
            best_y = y

    return best_ratio, best_x, best_y


def brute_force_consistency(lam: float, b: int, x_mult: int = 4):
    """Max ratio over x in [1, x_mult*b] with a perfect predictor (y = x).

    Returns (best_ratio, best_x).
    """
    x_max = x_mult * b
    x_arr = np.arange(1, x_max + 1, dtype=np.int64)
    opt_arr = np.minimum(x_arr, b).astype(float)

    tau_arr = tau_vec(x_arr, lam, b)
    cost_arr = np.where(x_arr < tau_arr, x_arr, (tau_arr - 1) + b).astype(float)
    ratio_arr = cost_arr / opt_arr

    idx = int(np.argmax(ratio_arr))
    return float(ratio_arr[idx]), int(x_arr[idx])
