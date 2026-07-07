"""Full-batch DP-GD with per-example gradient clipping, Gaussian noise, and an
optional canary contribution (the add/remove-one-record neighboring pair used
by the privacy audit).

Update rule, once per step t:
  1. Compute per-example gradients on the *full* dataset at the current theta.
  2. Clip each per-example gradient to L2 norm <= C.
  3. Sum (NOT average) the clipped per-example gradients -> S.
  4. If canary_in: add a fixed canary contribution C * e1 to S (a vector of
     exactly norm C along the first coordinate). This is present only in the
     "IN" world; omitting it in the "OUT" world models removing one record
     whose gradient contribution is exactly the canary vector, so the L2
     sensitivity of S between the two worlds is exactly C.
  5. Add Gaussian noise ~ N(0, (sigma*C)^2 I) to the (possibly canary-added) sum.
  6. theta <- theta - lr * noisy_sum.
"""
from __future__ import annotations

import numpy as np

from .model import per_example_gradients


def clip_gradients(grads: np.ndarray, C: float) -> np.ndarray:
    """Clip each row of grads (per-example gradients) to L2 norm <= C."""
    norms = np.linalg.norm(grads, axis=1)
    # Avoid division by zero for an all-zero gradient row.
    scale = np.minimum(1.0, C / np.maximum(norms, 1e-300))
    return grads * scale[:, None]


def train_dpgd(
    X: np.ndarray,
    y: np.ndarray,
    theta0: np.ndarray,
    T: int,
    C: float,
    sigma: float,
    lr: float,
    canary_in: bool,
    rng: np.random.Generator,
    return_trajectory: bool = False,
):
    """Run T steps of full-batch canary DP-GD from theta0.

    Args:
        X, y: full dataset (used in full-batch at every step).
        theta0: (d,) initial parameter vector.
        T: number of composed steps.
        C: per-example clipping norm (also the mechanism's L2 sensitivity).
        sigma: noise multiplier; per-step noise std is sigma * C.
        lr: learning rate.
        canary_in: whether the fixed canary contribution C*e1 is added to the
            gradient sum at every step ("IN" world) or omitted ("OUT" world).
        rng: explicitly-seeded np.random.Generator; the only source of
            randomness in this function (no global numpy random state is
            ever touched).
        return_trajectory: if True, also return the (T+1, d) array of all
            thetas visited (including theta0).

    Returns:
        theta_final (d,) if return_trajectory is False, else
        (theta_final, trajectory) where trajectory has shape (T+1, d).
    """
    d = theta0.shape[0]
    theta = theta0.astype(np.float64).copy()
    trajectory = [theta.copy()] if return_trajectory else None

    canary_vec = np.zeros(d)
    canary_vec[0] = C

    for _ in range(T):
        grads = per_example_gradients(X, y, theta)
        clipped = clip_gradients(grads, C)
        S = clipped.sum(axis=0)

        if canary_in:
            S = S + canary_vec

        noise = rng.normal(loc=0.0, scale=sigma * C, size=d)
        noisy_sum = S + noise

        theta = theta - lr * noisy_sum
        if return_trajectory:
            trajectory.append(theta.copy())

    if return_trajectory:
        return theta, np.array(trajectory)
    return theta
