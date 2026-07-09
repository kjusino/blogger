"""Boolean function families over {-1, +1}^n.

Convention: inputs and outputs live in {-1, +1}, with -1 read as "False"
and +1 as "True" (the standard analysis-of-Boolean-functions convention).
Every function exposes a fast, vectorized `evaluate_batch` so noise
sensitivity and influence can be estimated by Monte Carlo without ever
materializing a truth table, plus (where one exists) an exact closed form
for influence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class BooleanFunction:
    """Base class for a Boolean function f: {-1,+1}^n -> {-1,+1}."""

    n: int

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        """x has shape (num_samples, n) with entries in {-1, +1}.

        Returns shape (num_samples,) with entries in {-1, +1}.
        """
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> int:
        return int(self.evaluate_batch(x.reshape(1, -1))[0])


@dataclass
class ParityFunction(BooleanFunction):
    """f(x) = prod_i x_i. Every coordinate has influence exactly 1."""

    n: int

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        return np.prod(x, axis=1)


@dataclass
class MajorityFunction(BooleanFunction):
    """f(x) = sign(sum_i x_i). Requires odd n so there are no ties."""

    n: int

    def __post_init__(self) -> None:
        if self.n % 2 == 0:
            raise ValueError("MajorityFunction requires odd n (no ties allowed)")

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        return np.sign(np.sum(x, axis=1)).astype(np.int64)


@dataclass
class TribesFunction(BooleanFunction):
    """OR of `s` tribes, each an AND of `w` variables. n = w * s.

    The classic Ben-Or-Linial construction used to show the KKL influence
    bound Omega(log(n)/n) is tight up to a log-log factor.
    """

    w: int
    s: int
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = self.w * self.s

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        tribes = x.reshape(x.shape[0], self.s, self.w)
        tribe_and = np.all(tribes == 1, axis=2)  # True <-> tribe AND = +1
        formula_or = np.any(tribe_and, axis=1)  # True <-> some tribe fires
        return np.where(formula_or, 1, -1).astype(np.int64)


@dataclass
class RandomDNFFunction(BooleanFunction):
    """OR of `m` random terms, each an AND of `k` literals on random variables.

    No closed-form influence/noise-sensitivity theorem is known for this
    family in general; it is included as the exploratory arm of the
    experiment, interpolating between Parity-like (k=1) and Tribes/
    Majority-like behavior as k grows relative to log2(n).
    """

    n: int
    k: int
    m: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # term_vars[j]: the k variable indices in term j
        self.term_vars = np.array(
            [rng.choice(self.n, size=self.k, replace=False) for _ in range(self.m)]
        )
        # term_signs[j]: literal polarity (+1 = x_i, -1 = NOT x_i) per variable in term j
        self.term_signs = rng.choice([-1, 1], size=(self.m, self.k))

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        num_samples = x.shape[0]
        term_results = np.ones((num_samples, self.m), dtype=bool)
        for j in range(self.m):
            literals = x[:, self.term_vars[j]] * self.term_signs[j]  # +1 iff literal true
            term_results[:, j] = np.all(literals == 1, axis=1)
        formula_or = np.any(term_results, axis=1)
        return np.where(formula_or, 1, -1).astype(np.int64)


def majority_influence_exact(n: int) -> float:
    """Inf_i(Maj_n) = Pr[the other n-1 coordinates are tied] = C(n-1, (n-1)/2) / 2^(n-1).

    Computed in log-space via lgamma so it stays accurate for n in the
    hundreds of thousands.
    """
    if n % 2 == 0:
        raise ValueError("majority_influence_exact requires odd n")
    m = n - 1
    log_central_binom = math.lgamma(m + 1) - 2 * math.lgamma(m // 2 + 1)
    return math.exp(log_central_binom - m * math.log(2))


def tribes_influence_exact(w: int, s: int) -> float:
    """Closed form for the (symmetric) per-coordinate influence of Tribes(w, s).

    A variable i is pivotal iff (a) the other w-1 variables in its tribe are
    all true, AND (b) every other tribe is false:
        Inf_i = 2^-(w-1) * (1 - 2^-w)^(s-1)
    """
    p_tribe_true = 2.0 ** (-w)
    return 2.0 ** (-(w - 1)) * (1 - p_tribe_true) ** (s - 1)
