"""Bandit algorithms: EXP3 (adversarial) and UCB1 (stochastic).

Both expose the same two-method interface so they can be driven by a
common replay loop in ``src/regret.py``:

    select_arm() -> int
    update(arm: int, reward: float) -> None
"""

import math

import numpy as np


class EXP3:
    """Exponential-weight algorithm for Exploration and Exploitation.

    Auer, Cesa-Bianchi, Freund, Schapire (2002), "The Nonstochastic
    Multiarmed Bandit Problem", SIAM J. Comput. 32(1).

    Uses the fixed-horizon tuning gamma = min(1, sqrt(K ln K / ((e-1) T)))
    from Corollary 3.2 of that paper, which yields the expected weak-regret
    bound

        E[regret_T] <= 2 * sqrt(e - 1) * sqrt(K * T * ln K).

    Rewards passed to ``update`` must lie in [0, 1].
    """

    def __init__(self, n_arms, horizon, rng, gamma=None):
        if n_arms < 2:
            raise ValueError("EXP3 requires at least 2 arms")
        self.k = n_arms
        self.rng = rng
        if gamma is None:
            gamma = min(1.0, math.sqrt(self.k * math.log(self.k) / ((math.e - 1) * horizon)))
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        self.gamma = gamma
        self.weights = np.ones(self.k, dtype=np.float64)
        self._last_probs = None
        self._last_arm = None

    def probabilities(self):
        total = self.weights.sum()
        return (1.0 - self.gamma) * (self.weights / total) + self.gamma / self.k

    def select_arm(self):
        probs = self.probabilities()
        arm = int(self.rng.choice(self.k, p=probs))
        self._last_probs = probs
        self._last_arm = arm
        return arm

    def update(self, arm, reward):
        if arm != self._last_arm:
            raise ValueError("update() must follow select_arm() for the same arm")
        if not 0.0 <= reward <= 1.0:
            raise ValueError("EXP3 rewards must lie in [0, 1]")
        prob = self._last_probs[arm]
        estimated_reward = reward / prob
        self.weights[arm] *= math.exp(self.gamma * estimated_reward / self.k)
        # Renormalize to prevent overflow on long horizons; probabilities()
        # is scale-invariant in the weights so this changes nothing else.
        if self.weights.max() > 1e100:
            self.weights /= self.weights.max()


class UCB1:
    """Upper Confidence Bound algorithm for stochastic bandits.

    Auer, Cesa-Bianchi, Fischer (2002), "Finite-time Analysis of the
    Multiarmed Bandit Problem", Machine Learning 47.

    Plays every arm once, then plays argmax_i mean_i + sqrt(2 ln t / n_i).
    Theorem 1 of that paper bounds expected regret by

        sum_{i: Delta_i > 0} (8 ln T / Delta_i) + (1 + pi^2 / 3) * Delta_i.
    """

    def __init__(self, n_arms):
        if n_arms < 2:
            raise ValueError("UCB1 requires at least 2 arms")
        self.k = n_arms
        self.counts = np.zeros(self.k, dtype=np.int64)
        self.sums = np.zeros(self.k, dtype=np.float64)
        self.t = 0
        self._last_arm = None

    def select_arm(self):
        self.t += 1
        unplayed = np.flatnonzero(self.counts == 0)
        if unplayed.size > 0:
            arm = int(unplayed[0])
        else:
            means = self.sums / self.counts
            bonus = np.sqrt(2.0 * math.log(self.t) / self.counts)
            arm = int(np.argmax(means + bonus))
        self._last_arm = arm
        return arm

    def update(self, arm, reward):
        if arm != self._last_arm:
            raise ValueError("update() must follow select_arm() for the same arm")
        self.counts[arm] += 1
        self.sums[arm] += reward
