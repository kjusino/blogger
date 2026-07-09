"""Reward environments.

Both environments precompute a full (T, K) reward table with their own
RNG, independent of any algorithm's random choices, before play starts.
This is exactly the "oblivious adversary" formalization used in the
adversarial-bandit literature: the *distribution generating* rewards is
fixed in advance without observing the algorithm's plays, even when the
realized reward values are themselves randomized. A bandit algorithm only
ever observes ``table[t, arm_played]`` — never the other K-1 entries of
that row — via ``src/regret.run_bandit``.
"""

import numpy as np


class StochasticBernoulliEnv:
    """Stationary i.i.d. Bernoulli arms with fixed means."""

    def __init__(self, means, rng):
        means = np.asarray(means, dtype=np.float64)
        if np.any(means < 0.0) or np.any(means > 1.0):
            raise ValueError("means must lie in [0, 1]")
        self.means = means
        self.k = len(means)
        self.rng = rng

    def reward_table(self, horizon):
        draws = self.rng.random((horizon, self.k))
        return (draws < self.means[np.newaxis, :]).astype(np.float64)


class SwitchingBernoulliEnv:
    """Non-stationary Bernoulli arms: the "good" arm rotates cyclically.

    T rounds are split into equal segments of length ``segment_length``.
    Within segment s, the good arm is ``s % n_arms`` (a fixed, deterministic
    cyclic schedule decided before play starts — oblivious to any
    algorithm). The good arm's mean is ``0.5 + delta / 2``; all other arms
    are ``0.5 - delta / 2``. By symmetry every arm is "good" for exactly
    a 1/K share of rounds, so the best-fixed-arm-in-hindsight benchmark is
    close to tied across arms and no policy that *commits to a single arm*
    can beat it by more than realization noise — but a policy that tracks
    the current good arm across segments can (see README).
    """

    def __init__(self, n_arms, segment_length, delta, rng):
        if n_arms < 2:
            raise ValueError("need at least 2 arms")
        if not 0.0 < delta <= 1.0:
            raise ValueError("delta must lie in (0, 1]")
        if segment_length < 1:
            raise ValueError("segment_length must be >= 1")
        self.k = n_arms
        self.segment_length = segment_length
        self.delta = delta
        self.rng = rng

    def good_arm_schedule(self, horizon):
        segment_idx = np.arange(horizon) // self.segment_length
        return segment_idx % self.k

    def mean_table(self, horizon):
        schedule = self.good_arm_schedule(horizon)
        means = np.full((horizon, self.k), 0.5 - self.delta / 2.0, dtype=np.float64)
        means[np.arange(horizon), schedule] = 0.5 + self.delta / 2.0
        return means

    def reward_table(self, horizon):
        means = self.mean_table(horizon)
        draws = self.rng.random((horizon, self.k))
        return (draws < means).astype(np.float64)
