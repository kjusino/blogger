"""A generic Benettin-style QR Lyapunov-spectrum estimator.

Works on any discrete-time map `x_{k+1} = step_fn(x_k)` given either an
analytic Jacobian callable `jacobian_fn(x) -> (dim, dim) array` or using the
finite-difference Jacobian fallback provided here. This makes it usable both
for:
  (a) the "true" Lorenz flow map advanced by a fixed dt (composed with the
      RK4 integrator in dynamics.py), and
  (b) an arbitrary trained neural-network surrogate's forward pass
      (surrogate.py), by finite-differencing the network's output.

Algorithm (Benettin et al. 1980 / standard QR method):
  1. Advance the state under the map.
  2. Propagate an orthonormal basis of tangent vectors Q through the
     map's Jacobian at each step: Q <- J(x) @ Q.
  3. Periodically re-orthonormalize Q via QR decomposition (with a sign
     convention that fixes R's diagonal to be positive, so the QR
     factorization -- and hence log|R_ii| -- is well-defined/unique).
  4. Accumulate log|R_ii| across all steps; after `n_iters` steps (each
     representing `dt` units of time), the i-th Lyapunov exponent is
     estimated as sum(log|R_ii|) / (n_iters * dt).
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def finite_difference_jacobian(map_fn: Callable[[np.ndarray], np.ndarray],
                                x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of `map_fn` at point `x`.

    map_fn: R^d -> R^d. Returns a (d, d) matrix J with J[i, j] = d f_i / d x_j.
    """
    x = np.asarray(x, dtype=float)
    dim = x.shape[0]
    J = np.zeros((dim, dim))
    for j in range(dim):
        step = np.zeros(dim)
        step[j] = eps
        f_plus = map_fn(x + step)
        f_minus = map_fn(x - step)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
    return J


def lyapunov_spectrum(step_fn: Callable[[np.ndarray], np.ndarray],
                       x0: np.ndarray,
                       n_iters: int,
                       jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                       dt: float = 1.0,
                       warmup: int = 0,
                       renorm_interval: int = 1,
                       fd_eps: float = 1e-6) -> np.ndarray:
    """Estimate the full Lyapunov exponent spectrum of a discrete map.

    Parameters
    ----------
    step_fn : x_k -> x_{k+1}, the discrete map to analyze.
    x0 : initial state (1-D array of length `dim`).
    n_iters : number of map iterations used for the estimate (after warmup).
    jacobian_fn : optional analytic Jacobian of `step_fn` at a state; if not
        given, a finite-difference Jacobian of `step_fn` is used instead.
    dt : the amount of "time" one application of step_fn represents (1.0 for
        genuinely discrete maps like the logistic map; the physical
        integration dt for a flow map sampled at fixed intervals, e.g. 0.01
        for the Lorenz flow map used throughout this project).
    warmup : number of map iterations to discard first, letting the
        trajectory settle onto the attractor before accumulating statistics.
    renorm_interval : how many map steps to take between QR
        renormalizations of the tangent basis. 1 is safest numerically
        (avoids overflow of the propagated Jacobian products) and is the
        default; larger values trade a little accuracy for speed.
    fd_eps : step size used by the finite-difference Jacobian fallback.

    Returns
    -------
    exponents : array of length `dim`, sorted descending (lambda_1 first).
    """
    x = np.asarray(x0, dtype=float).copy()
    dim = x.shape[0]

    if jacobian_fn is None:
        jac = lambda state: finite_difference_jacobian(step_fn, state, eps=fd_eps)
    else:
        jac = jacobian_fn

    for _ in range(warmup):
        x = step_fn(x)

    Q = np.eye(dim)
    log_sums = np.zeros(dim)
    steps_since_renorm = 0
    accumulated_steps = 0

    for k in range(n_iters):
        J = jac(x)
        Q = J @ Q
        x = step_fn(x)
        steps_since_renorm += 1

        is_last = (k == n_iters - 1)
        if steps_since_renorm >= renorm_interval or is_last:
            Q_new, R = np.linalg.qr(Q)
            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1.0
            Q_new = Q_new * signs  # broadcasts over columns
            R = (signs[:, None] * R)
            diag = np.abs(np.diag(R))
            # Guard against a numerically-exact zero (would give -inf log);
            # this only happens for a wholly degenerate/non-invertible map.
            diag = np.clip(diag, 1e-300, None)
            log_sums += np.log(diag)
            Q = Q_new
            accumulated_steps += steps_since_renorm
            steps_since_renorm = 0

    total_time = accumulated_steps * dt
    exponents = log_sums / total_time
    return np.sort(exponents)[::-1]
