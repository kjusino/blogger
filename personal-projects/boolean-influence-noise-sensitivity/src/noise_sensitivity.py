"""Monte Carlo estimator for Boolean-function noise sensitivity.

NS_delta(f) = Pr[f(x) != f(y)], where x ~ Uniform({-1,+1}^n) and y is a
delta-noisy copy of x: each coordinate of y independently equals the
corresponding coordinate of x with probability (1 - delta), and is
rerandomized (uniform +-1) with probability delta.

This only requires evaluating f, so it works for any n at any width
without ever building a truth table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .functions import BooleanFunction


@dataclass
class NoiseSensitivityResult:
    delta: float
    estimate: float
    stderr: float
    n_samples: int


def monte_carlo_noise_sensitivity(
    f: BooleanFunction,
    delta: float,
    n_samples: int,
    rng: np.random.Generator,
) -> NoiseSensitivityResult:
    if not 0.0 <= delta <= 1.0:
        raise ValueError("delta must be in [0, 1]")

    n = f.n
    x = rng.choice([-1, 1], size=(n_samples, n))
    rerandomize_mask = rng.random(size=(n_samples, n)) < delta
    fresh = rng.choice([-1, 1], size=(n_samples, n))
    y = np.where(rerandomize_mask, fresh, x)

    disagree = (f.evaluate_batch(x) != f.evaluate_batch(y)).astype(np.float64)
    p_hat = float(np.mean(disagree))
    stderr = float(np.std(disagree, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else 0.0
    return NoiseSensitivityResult(delta=delta, estimate=p_hat, stderr=stderr, n_samples=n_samples)


def majority_sheppard_limit(delta: float) -> float:
    """lim_{n->infinity} NS_delta(Maj_n) = arccos(1 - delta) / pi.

    By the CLT, (sum x_i, sum y_i) / sqrt(n) for a delta-noisy pair (x, y)
    converges to a pair of standard Gaussians with correlation rho = 1 -
    delta, and Sheppard's formula gives Pr[sign(g1) != sign(g2)] =
    arccos(rho) / pi for correlated Gaussians. This is the "typical
    balanced function" baseline that BKS's Tribes construction beats.
    """
    return math.acos(1 - delta) / math.pi


def parity_noise_sensitivity_exact(n: int, delta: float) -> float:
    """Closed form: NS_delta(Parity_n) = (1 - (1 - delta)^n) / 2.

    Parity has a single nonzero Fourier coefficient, on the full set [n], so
    Stab_rho(Parity_n) = rho^n with rho = E[x_i y_i] = 1 - delta under the
    "resample each coordinate independently w.p. delta" noise model used by
    `monte_carlo_noise_sensitivity`, giving NS_delta = (1 - Stab_rho) / 2.
    Used as a ground truth to validate the Monte Carlo estimator.
    """
    return (1 - (1 - delta) ** n) / 2
