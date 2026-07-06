"""Discrete probability distributions over a finite domain [0, n) used to
generate uniform and epsilon-far-from-uniform test instances.

All "far" constructions below are exact: the returned distribution has total
variation (TV) distance to the uniform distribution equal to `epsilon` up to
floating point error, not merely approximately epsilon. This matters because
the theoretical sample-complexity bounds we test against (see theory.py) are
stated in terms of exact TV distance.
"""

from __future__ import annotations

import numpy as np


def uniform(n: int) -> np.ndarray:
    """The uniform distribution over n elements."""
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(n, 1.0 / n)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions over the same domain."""
    return 0.5 * float(np.sum(np.abs(p - q)))


def paired_perturbation(n: int, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """The classical "paired" hard instance for uniformity testing (Paninski 2008).

    Partitions the domain into n/2 disjoint pairs. Within each pair one element
    gets mass (1 + 2*epsilon)/n and the other gets (1 - 2*epsilon)/n, so each
    pair still sums to the uniform pair-mass 2/n. This construction is the
    standard lower-bound instance showing uniformity testing requires
    Omega(sqrt(n)/epsilon^2) samples: collision statistics on this family are
    (in expectation) the hardest to distinguish from uniform.

    Requires n even and 0 <= epsilon <= 0.5 (so masses stay non-negative).
    """
    if n % 2 != 0:
        raise ValueError("paired_perturbation requires an even n")
    if not (0.0 <= epsilon <= 0.5):
        raise ValueError("epsilon must be in [0, 0.5]")
    high = (1.0 + 2.0 * epsilon) / n
    low = (1.0 - 2.0 * epsilon) / n
    pattern = np.empty(n)
    pattern[0::2] = high
    pattern[1::2] = low
    # Randomly relabel domain elements each draw so repeated trials at the
    # same (n, epsilon) don't all share the exact same support structure,
    # while exact TV distance to uniform (which is permutation-invariant)
    # is unaffected.
    perm = rng.permutation(n)
    return pattern[perm]


def block_perturbation(
    n: int, epsilon: float, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """A "concentrated" far instance: a block of `block_size` elements absorbs
    all of the excess probability mass, the remaining n - block_size elements
    share the deficit.

    p[i] = 1/n + epsilon/block_size            for i in the block
    p[i] = 1/n - epsilon/(n - block_size)       otherwise

    TV distance to uniform is exactly epsilon. With block_size = 1 this is the
    "single heavy element" instance; larger block_size spreads the same excess
    mass over more elements, which collision-based testers generally find
    easier to detect (concentration in a *few* elements raises the collision
    probability more per unit of TV distance).

    Requires 1 <= block_size < n and epsilon small enough that both masses
    stay non-negative.
    """
    if not (1 <= block_size < n):
        raise ValueError("block_size must satisfy 1 <= block_size < n")
    high = 1.0 / n + epsilon / block_size
    low = 1.0 / n - epsilon / (n - block_size)
    if low < 0:
        raise ValueError("epsilon too large for this block_size: negative mass")
    p = np.full(n, low)
    block_idx = rng.choice(n, size=block_size, replace=False)
    p[block_idx] = high
    return p


FAMILIES = {
    "paired": lambda n, eps, rng: paired_perturbation(n, eps, rng),
    "single_heavy": lambda n, eps, rng: block_perturbation(n, eps, 1, rng),
    "block_quarter": lambda n, eps, rng: block_perturbation(
        n, eps, max(1, n // 4), rng
    ),
}
