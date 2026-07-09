"""Exact and Monte Carlo estimators for Boolean-function influence.

Inf_i(f) = Pr_x[f(x) != f(x with coordinate i flipped)],  x ~ Uniform({-1,+1}^n)
I(f)     = sum_i Inf_i(f)          (total influence / average sensitivity)
Var(f)   = 1 - E[f(x)]^2           (for f: {-1,+1}^n -> {-1,+1})
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .functions import BooleanFunction

MAX_EXACT_N = 20  # 2**20 ~= 1e6 rows; brute force is fine up to here


@dataclass
class InfluenceResult:
    per_coordinate: np.ndarray  # shape (n,)
    total_influence: float
    variance: float
    max_influence: float
    method: str


def exact_truth_table(f: BooleanFunction) -> InfluenceResult:
    """Brute-force influence over the full 2^n truth table. Only for small n."""
    n = f.n
    if n > MAX_EXACT_N:
        raise ValueError(f"n={n} too large for exact enumeration (max {MAX_EXACT_N})")

    all_inputs = np.array(
        [[1 if (idx >> bit) & 1 else -1 for bit in range(n)] for idx in range(2**n)],
        dtype=np.int64,
    )
    f_vals = f.evaluate_batch(all_inputs)

    per_coord = np.empty(n)
    for i in range(n):
        flipped = all_inputs.copy()
        flipped[:, i] *= -1
        f_flipped = f.evaluate_batch(flipped)
        per_coord[i] = np.mean(f_vals != f_flipped)

    mean_f = np.mean(f_vals)
    variance = 1 - mean_f**2
    return InfluenceResult(
        per_coordinate=per_coord,
        total_influence=float(np.sum(per_coord)),
        variance=float(variance),
        max_influence=float(np.max(per_coord)),
        method="exact",
    )


def monte_carlo_influence(
    f: BooleanFunction,
    n_samples: int,
    rng: np.random.Generator,
    coordinates: np.ndarray | None = None,
) -> InfluenceResult:
    """Estimate influence for a subset of coordinates (default: all) by sampling.

    Total influence is extrapolated as mean(per-coordinate estimate) * n when
    only a subset of coordinates is sampled (valid for exchangeable/symmetric
    function families such as Majority and Tribes; for asymmetric families
    such as RandomDNF, pass coordinates=np.arange(n) for an exact sum).
    """
    n = f.n
    coords = np.arange(n) if coordinates is None else coordinates

    x = rng.choice([-1, 1], size=(n_samples, n))
    f_x = f.evaluate_batch(x)
    mean_f = np.mean(f_x)
    variance = 1 - mean_f**2

    per_coord = np.empty(len(coords))
    for idx, i in enumerate(coords):
        flipped = x.copy()
        flipped[:, i] *= -1
        f_flipped = f.evaluate_batch(flipped)
        per_coord[idx] = np.mean(f_x != f_flipped)

    if coordinates is None or len(coords) == n:
        total = float(np.sum(per_coord))
    else:
        total = float(np.mean(per_coord) * n)

    return InfluenceResult(
        per_coordinate=per_coord,
        total_influence=total,
        variance=float(variance),
        max_influence=float(np.max(per_coord)),
        method="monte_carlo",
    )
