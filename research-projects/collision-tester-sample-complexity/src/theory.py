"""Closed-form (up to constants) sample-complexity predictions from the
property-testing literature, used only as a reference against which we
compare empirically fitted scaling exponents. We never claim to recover the
exact constants — only the scaling exponent in n (0.5 for the collision
tester, 1.0 for the naive learner) and, for the collision tester's
threshold, the exact collision-probability gap realized by the paired
construction.
"""

from __future__ import annotations

import math


def collision_tester_predicted_m(n: int, epsilon: float, const: float = 1.0) -> float:
    """Theta(sqrt(n)/epsilon^2) sample complexity of the collision tester."""
    return const * math.sqrt(n) / epsilon ** 2


def naive_learner_predicted_m(n: int, epsilon: float, const: float = 1.0) -> float:
    """Theta(n/epsilon^2) sample complexity of learning-then-testing."""
    return const * n / epsilon ** 2


PREDICTED_M = {
    "collision": collision_tester_predicted_m,
    "naive_learner": naive_learner_predicted_m,
}

PREDICTED_EXPONENT = {
    "collision": 0.5,
    "naive_learner": 1.0,
}


def paired_collision_probability(n: int, epsilon: float) -> float:
    """Exact sum_i p_i^2 for the paired_perturbation construction: (1+4*eps^2)/n.

    Derivation: half the n elements carry mass (1+2*eps)/n, half carry
    (1-2*eps)/n. sum p_i^2 = (n/2)*[(1+2*eps)/n]^2 + (n/2)*[(1-2*eps)/n]^2
    = (1/n) * (1 + 4*eps^2).
    """
    return (1.0 + 4.0 * epsilon ** 2) / n
