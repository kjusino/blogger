"""Two testers for the hypothesis H0: "samples are drawn from Uniform(n)" vs
H1: "the source distribution has total variation distance >= epsilon from
uniform".

1. `collision_tester` — the sublinear Goldreich-Ron / collision-counting
   tester. It never estimates the distribution; it only counts how often two
   samples land on the same domain element and compares that count to the
   count expected under uniformity. Known sample complexity: O(sqrt(n)/eps^2).

2. `naive_learner_tester` — estimates the full empirical histogram and
   measures its L1 distance to uniform. This is what you'd do if you actually
   learned the distribution first. Known sample complexity: Theta(n/eps^2).

Both testers are calibrated only from (n, epsilon) — never from which
distribution family generated the samples — exactly as a real tester would
be, since the family is unknown at test time.
"""

from __future__ import annotations

import numpy as np


def collision_probability(p: np.ndarray) -> float:
    """Exact collision probability sum_i p_i^2 of a (known) distribution.

    Used only for validation / ground truth in tests and figures, never by
    the testers themselves (which only see samples).
    """
    return float(np.sum(p ** 2))


def collision_statistic(samples: np.ndarray, n: int) -> int:
    """Number of unordered colliding pairs among the given samples."""
    counts = np.bincount(samples, minlength=n)
    return int(np.sum(counts * (counts - 1) // 2))


def collision_tester(samples: np.ndarray, n: int, epsilon: float) -> bool:
    """Goldreich-Ron collision tester. Returns True to reject H0 (declare "far").

    Threshold is the midpoint, in collision probability, between the null
    (1/n) and the tight worst-case alternative (1 + 4*eps^2)/n realized by
    the `paired_perturbation` construction in distributions.py:

        threshold_prob = (1 + 2*eps^2) / n
    """
    m = samples.shape[0]
    if m < 2:
        raise ValueError("collision_tester needs at least 2 samples")
    y = collision_statistic(samples, n)
    total_pairs = m * (m - 1) / 2.0
    threshold = total_pairs * (1.0 + 2.0 * epsilon ** 2) / n
    return y > threshold


def naive_learner_tester(samples: np.ndarray, n: int, epsilon: float) -> bool:
    """Learn the empirical histogram, threshold on its L1 distance to uniform.

    A distribution at TV distance epsilon from uniform has L1 distance
    2*epsilon (L1 = 2 * TV always). We reject H0 if the empirical L1 distance
    exceeds the midpoint `epsilon` between 0 (perfectly uniform) and 2*epsilon
    (the alternative).
    """
    m = samples.shape[0]
    counts = np.bincount(samples, minlength=n)
    p_hat = counts / m
    l1 = float(np.sum(np.abs(p_hat - 1.0 / n)))
    return l1 > epsilon


TESTERS = {
    "collision": collision_tester,
    "naive_learner": naive_learner_tester,
}
