"""Membership / equivalence oracles for active automata learning, including a
persistent-noise membership oracle and a Hoeffding-bound redundancy wrapper
that recovers a target per-query error probability by majority-vote repetition.
"""

from __future__ import annotations

import math
import random

from .dfa import DFA, equivalence_counterexample


class QueryBudgetExceeded(Exception):
    pass


class MembershipOracle:
    """Exact membership oracle for a target DFA. Counts raw calls."""

    def __init__(self, target: DFA):
        self.target = target
        self.raw_queries = 0

    def query(self, word) -> bool:
        self.raw_queries += 1
        return self.target.accepts(word)


class NoisyMembershipOracle:
    """Wraps a MembershipOracle; each raw call independently flips the true
    answer with probability `noise_rate` (persistent/classification noise,
    not correlated across repeated calls on the same word)."""

    def __init__(self, target: DFA, noise_rate: float, rng: random.Random):
        if not (0.0 <= noise_rate < 0.5):
            raise ValueError("noise_rate must be in [0, 0.5) for majority voting to help")
        self.inner = MembershipOracle(target)
        self.noise_rate = noise_rate
        self.rng = rng

    @property
    def raw_queries(self) -> int:
        return self.inner.raw_queries

    def query_once(self, word) -> bool:
        truth = self.inner.query(word)
        if self.rng.random() < self.noise_rate:
            return not truth
        return truth


def hoeffding_repetitions(noise_rate: float, delta: float) -> int:
    """Minimum odd k such that a majority vote over k iid Bernoulli(1-noise_rate)
    calls is wrong with probability <= delta, via the Hoeffding bound

        P(majority wrong) <= exp(-2 k (1/2 - p)^2)

    Solving exp(-2 k (1/2-p)^2) <= delta for k gives
        k >= ln(1/delta) / (2 (1/2 - p)^2).
    """
    if noise_rate <= 0.0:
        return 1
    margin = 0.5 - noise_rate
    k = math.ceil(math.log(1.0 / delta) / (2.0 * margin * margin))
    if k % 2 == 0:
        k += 1
    return max(1, k)


class RedundantMembershipOracle:
    """Answers each distinct membership query by repeating the noisy oracle
    `repetitions` times (fixed, or computed per-call by `repetitions_fn`) and
    taking a majority vote. Caches distinct queries so re-asking the same word
    (as L* does heavily) doesn't re-spend the query budget."""

    def __init__(self, noisy: NoisyMembershipOracle, repetitions: int = None,
                 repetitions_fn=None):
        if (repetitions is None) == (repetitions_fn is None):
            raise ValueError("pass exactly one of repetitions, repetitions_fn")
        self.noisy = noisy
        self.repetitions = repetitions
        self.repetitions_fn = repetitions_fn
        self._cache = {}
        self.distinct_queries = 0

    def _k(self) -> int:
        return self.repetitions if self.repetitions is not None else self.repetitions_fn(self.noisy.noise_rate)

    def query(self, word) -> bool:
        key = tuple(word)
        if key in self._cache:
            return self._cache[key]
        self.distinct_queries += 1
        k = self._k()
        votes = sum(1 for _ in range(k) if self.noisy.query_once(word))
        result = votes * 2 > k
        self._cache[key] = result
        return result

    @property
    def raw_queries(self) -> int:
        return self.noisy.raw_queries


class ExactEquivalenceOracle:
    """Ground-truth equivalence oracle used to evaluate learned hypotheses.

    Membership noise is the variable under study; the equivalence oracle is
    intentionally kept exact so that learning failures can be attributed
    solely to membership-query noise (see README, "Scope and limitations").
    """

    def __init__(self, target: DFA):
        self.target = target
        self.num_queries = 0

    def query(self, hypothesis: DFA):
        self.num_queries += 1
        return equivalence_counterexample(self.target, hypothesis)
