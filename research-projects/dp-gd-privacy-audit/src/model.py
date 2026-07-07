"""Manual logistic regression gradient (no autodiff / no ML framework).

Loss (mean over examples is NOT used here -- see dpgd.py which works with the
per-example gradient list and later the *sum*, never the average):

    L_i(theta) = -[ y_i * log(sigma(x_i . theta)) + (1-y_i) * log(1 - sigma(x_i . theta)) ]

Gradient of a single example's loss w.r.t. theta:

    grad_i(theta) = (sigma(x_i . theta) - y_i) * x_i
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def per_example_gradients(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Per-example gradient of the logistic-regression loss.

    Args:
        X: (n, d) features.
        y: (n,) labels in {0, 1}.
        theta: (d,) parameter vector.

    Returns:
        (n, d) array; row i is grad_i(theta) = (sigmoid(x_i . theta) - y_i) * x_i.
    """
    z = X @ theta  # (n,)
    p = sigmoid(z)  # (n,)
    residual = p - y  # (n,)
    return residual[:, None] * X  # (n, d)


def loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Mean logistic loss, used only for gradient-check testing (not in DP-GD)."""
    z = X @ theta
    p = sigmoid(z)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
