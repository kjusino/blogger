"""Samplers for the stochastic (Bayesian-optimal-lambda) study.

Two things are sampled:
  1. A true trip length x, from a configurable distribution over "days".
  2. A noisy prediction y of x, via a log-normal multiplicative noise model:
     y = round(x * exp(Z)), Z ~ Normal(0, sigma). sigma = 0 is a perfect
     predictor; larger sigma is a worse one. y is clipped to be >= 0.
"""

from __future__ import annotations

import numpy as np


def sample_x_lognormal(
    n: int, b: int, rng: np.random.Generator, median_days: float | None = None,
    sigma: float = 0.6,
) -> np.ndarray:
    """Discretized log-normal trip-length distribution.

    Centered (median) at `median_days`, defaulting to b itself so the buy
    decision is genuinely contested (trip length is comparable to the
    buy-out cost, rather than trivially always-short or always-long).
    """
    if median_days is None:
        median_days = float(b)
    z = rng.normal(0.0, sigma, size=n)
    x = median_days * np.exp(z)
    x = np.round(x).astype(np.int64)
    x = np.clip(x, 1, None)
    return x


def sample_x_mixture(
    n: int, b: int, rng: np.random.Generator, short_frac: float = 0.5,
    short_scale: float = 0.2, long_scale: float = 3.0,
) -> np.ndarray:
    """Mixture of a "short trip" component (exponential, mean
    short_scale*b, i.e. well under the buy-out cost) and a "long trip"
    component (b plus an exponential with mean long_scale*b, i.e. well
    past the buy-out cost)."""
    is_short = rng.random(n) < short_frac
    n_short = int(is_short.sum())
    n_long = n - n_short

    short_vals = rng.exponential(scale=short_scale * b, size=n_short)
    long_vals = b + rng.exponential(scale=long_scale * b, size=n_long)

    x = np.empty(n, dtype=float)
    x[is_short] = short_vals
    x[~is_short] = long_vals

    x = np.round(x).astype(np.int64)
    x = np.clip(x, 1, None)
    return x


def noisy_prediction(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """y = round(x * exp(Z)), Z ~ Normal(0, sigma); sigma = 0 => y = x exactly."""
    x = np.asarray(x, dtype=float)
    if sigma == 0.0:
        y = np.round(x).astype(np.int64)
    else:
        z = rng.normal(0.0, sigma, size=len(x))
        y = np.round(x * np.exp(z)).astype(np.int64)
    y = np.clip(y, 0, None)
    return y
