"""Monte Carlo estimation of expected competitive ratio under a stochastic
prediction-error model, and grid search for the empirically-optimal lambda.

Also implements a simple hand-derived closed-form heuristic
lambda*_approx(sigma) = exp(-c * sigma), fit from two empirical anchor
points (sigma = 0, where the true optimum is exactly lambda* = 1 since a
perfect predictor should be fully trusted -- Consistency(1, b) = 1 is
optimal -- and the largest sigma in the sweep). This heuristic is not
claimed to be theoretically optimal; it is compared honestly against the
Monte Carlo argmin in the README/results.

A methodological note on `argmin_lambda`: it draws ONE batch of (x, y)
samples per sigma and reuses it for every lambda in the grid ("common
random numbers"), rather than redrawing fresh samples per lambda. An
earlier version of this code redrew samples per lambda from a single
continuously-advancing RNG, so adjacent lambda estimates were being
compared against different noise realizations -- this adds sampling noise
on top of the true lambda-effect and can flip a close argmin. Sharing the
same sample batch across the whole lambda grid removes that source of
noise and makes the comparison between lambda values fair (see
tests/test_monte_carlo.py for a regression test and README.md's "bugs
found" note for the empirical impact this had at high sigma).
"""

from __future__ import annotations

import numpy as np

from .algorithm import tau_vec, cost, opt
from .predictor import noisy_prediction


def expected_ratio_given_samples(lam: float, b: int, x: np.ndarray, y: np.ndarray) -> float:
    """Monte Carlo expected ratio for a *fixed* batch of (x, y) samples.

    Splitting this out from expected_ratio() lets argmin_lambda() reuse the
    exact same sample batch across every lambda in its grid search (common
    random numbers), instead of drawing independent samples per lambda.
    """
    tau_arr = tau_vec(y, lam, b)
    cost_arr = cost(x, tau_arr, b).astype(float)
    opt_arr = opt(x, b).astype(float)
    # x_sampler always returns x >= 1, so opt_arr is always > 0 here.
    ratios = cost_arr / opt_arr
    return float(np.mean(ratios))


def expected_ratio(
    lam: float, b: int, sigma: float, n_samples: int,
    rng: np.random.Generator, x_sampler,
) -> float:
    """Monte Carlo estimate of E[cost(x, tau(y, lam, b), b) / OPT(x, b)]
    under x ~ x_sampler and y = noisy_prediction(x, sigma)."""
    x = x_sampler(n_samples, b, rng)
    y = noisy_prediction(x, sigma, rng)
    return expected_ratio_given_samples(lam, b, x, y)


def argmin_lambda(
    b: int, sigma: float, lambda_grid: np.ndarray, n_samples: int,
    rng: np.random.Generator, x_sampler,
):
    """Grid search over lambda_grid for the lambda minimizing the Monte
    Carlo expected ratio. Returns (best_lambda, ratios_array).

    Draws a single (x, y) sample batch and reuses it for every lambda in
    the grid (common random numbers) -- see the module docstring.
    """
    x = x_sampler(n_samples, b, rng)
    y = noisy_prediction(x, sigma, rng)
    ratios = np.array(
        [expected_ratio_given_samples(lam, b, x, y) for lam in lambda_grid]
    )
    idx = int(np.argmin(ratios))
    return float(lambda_grid[idx]), ratios


def lambda_star_approx(sigma: float, c: float) -> float:
    """Simple closed-form heuristic: lambda*_approx(sigma) = exp(-c*sigma).

    By construction lambda*_approx(0) = 1 for any c, matching the exact
    result that a perfect predictor (sigma = 0) should be fully trusted.
    """
    return float(np.exp(-c * sigma))


def fit_c(sigma_anchor: float, lambda_star_at_anchor: float) -> float:
    """Fit c from a single non-zero anchor point (sigma_anchor,
    lambda_star_at_anchor) by solving lambda*_approx(sigma_anchor) =
    exp(-c * sigma_anchor) for c. The sigma = 0 anchor is automatically
    satisfied (both sides equal 1) and carries no information about c, so
    it is not used here.
    """
    if sigma_anchor <= 0:
        raise ValueError("sigma_anchor must be > 0 to fit c")
    ls = min(max(lambda_star_at_anchor, 1e-6), 1.0)
    return float(-np.log(ls) / sigma_anchor)
