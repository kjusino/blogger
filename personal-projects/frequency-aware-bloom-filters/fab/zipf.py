"""Synthetic Zipfian popularity model for a universe of keys.

``popularity_weights`` gives every key in a universe of size N a query
weight w_i proportional to rank(i)^-s (the Zipf skew parameter s controls
how concentrated traffic is on a few "hot" keys; s=0 is uniform, larger s
is more skewed). ``sample_stream`` draws a synthetic access log from this
distribution, used to train the Count-Min Sketch the way a real system
would train one from historical traffic.
"""
from __future__ import annotations

import numpy as np


def popularity_weights(num_keys: int, skew: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a length-``num_keys`` array of normalized Zipf(skew) weights.

    Ranks are randomly permuted (via ``rng``) so "popular" is not simply
    "low key index" -- this matters once we split keys into a random true
    set S independent of popularity.
    """
    if num_keys <= 0:
        raise ValueError("num_keys must be positive")
    ranks = np.arange(1, num_keys + 1, dtype=np.float64)
    raw = ranks ** (-skew)
    weights = raw / raw.sum()
    if rng is not None:
        rng.shuffle(weights)
    return weights


def sample_stream(weights: np.ndarray, length: int, rng: np.random.Generator) -> list[str]:
    """Draw ``length`` key ids (as strings "k{idx}") i.i.d. from ``weights``."""
    idx = rng.choice(len(weights), size=length, p=weights)
    return [f"k{i}" for i in idx]
