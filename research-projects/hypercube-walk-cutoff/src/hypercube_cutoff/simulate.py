"""Monte Carlo simulation of the actual random walk (not the algebraic shortcut).

Two simulators are provided:

- `simulate_bitvector_trajectories`: the literal process -- each of `num_trials`
  independent walkers holds an explicit length-n bit vector; at each step, a
  uniformly random coordinate is overwritten with a fresh fair coin flip.
  Used to validate that real sample paths behave as the lumping in
  `chain.py` predicts.
- `simulate_weight_trajectories`: simulates only the Hamming weight via the
  equivalent birth-death jump probabilities. Valid because of the exact
  lumping (see `chain.py` module docstring), and much cheaper for large n.

`empirical_tv_distance` turns a batch of sampled weights at a fixed time into
an estimated TV distance to Binomial(n, 1/2), with a bootstrap CI.
"""

from __future__ import annotations

import numpy as np

from .chain import stationary_weight_pmf


def simulate_bitvector_trajectories(n: int, t_max: int, num_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate `num_trials` independent bit-vector walkers for t_max steps.

    Returns the Hamming weight of each walker at every step, shape (t_max+1, num_trials).
    Intended for small-to-moderate n (memory is O(num_trials * n)); this is the
    literal process, not the weight-chain shortcut.
    """
    if t_max < 0:
        raise ValueError("t_max must be nonnegative")
    state = np.zeros((num_trials, n), dtype=np.int8)
    weights = np.empty((t_max + 1, num_trials), dtype=np.int64)
    weights[0] = state.sum(axis=1)
    trial_idx = np.arange(num_trials)
    for t in range(1, t_max + 1):
        coord = rng.integers(0, n, size=num_trials)
        newbit = rng.integers(0, 2, size=num_trials, dtype=np.int8)
        state[trial_idx, coord] = newbit
        weights[t] = state.sum(axis=1)
    return weights


def simulate_weight_trajectories(n: int, t_max: int, num_trials: int, rng: np.random.Generator,
                                  start_weight: int = 0) -> np.ndarray:
    """Simulate `num_trials` independent copies of the lumped weight chain.

    Returns weight of each walker at every step, shape (t_max+1, num_trials).
    Equivalent in law to `simulate_bitvector_trajectories`'s weight output, but
    O(num_trials) memory per step regardless of n, so usable for large n.
    """
    if t_max < 0:
        raise ValueError("t_max must be nonnegative")
    if not (0 <= start_weight <= n):
        raise ValueError("start_weight must be in [0, n]")
    w = np.full(num_trials, start_weight, dtype=np.int64)
    weights = np.empty((t_max + 1, num_trials), dtype=np.int64)
    weights[0] = w
    for t in range(1, t_max + 1):
        u = rng.random(num_trials)
        down_prob = w / (2 * n)
        up_prob = (n - w) / (2 * n)
        down = u < down_prob
        up = (~down) & (u < down_prob + up_prob)
        w = w - down.astype(np.int64) + up.astype(np.int64)
        weights[t] = w
    return weights


def empirical_tv_distance(weight_samples: np.ndarray, n: int, n_bootstrap: int = 0,
                           rng: np.random.Generator | None = None, ci: float = 0.95):
    """Empirical TV distance between the sampled weight distribution and
    Binomial(n, 1/2). If n_bootstrap > 0, also returns a percentile CI from
    bootstrap resampling of the samples.
    """
    weight_samples = np.asarray(weight_samples)
    pi = stationary_weight_pmf(n)
    counts = np.bincount(weight_samples, minlength=n + 1).astype(float)
    if counts.shape[0] != n + 1:
        raise ValueError("weight_samples contains a value outside [0, n]")
    emp_pmf = counts / counts.sum()
    tv = 0.5 * np.sum(np.abs(emp_pmf - pi))

    if n_bootstrap <= 0:
        return tv, None

    if rng is None:
        raise ValueError("rng is required when n_bootstrap > 0")
    m = weight_samples.shape[0]
    boot_tvs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        resample = rng.choice(weight_samples, size=m, replace=True)
        c = np.bincount(resample, minlength=n + 1).astype(float)
        p = c / c.sum()
        boot_tvs[b] = 0.5 * np.sum(np.abs(p - pi))
    alpha = 1 - ci
    lo, hi = np.percentile(boot_tvs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return tv, (lo, hi)
