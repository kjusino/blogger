"""Density evolution for (dv, dc)-regular LDPC ensembles over the BEC.

For the binary erasure channel, density evolution reduces to a scalar
recursion on the edge-perspective erasure probability x_l (the probability
that a variable-to-check message at iteration l is an erasure), for the
infinite-blocklength, cycle-free ensemble limit:

    x_0     = epsilon
    x_{l+1} = epsilon * (1 - (1 - x_l)^(dc-1))^(dv-1)

x_l is monotonically non-increasing in l for fixed epsilon, and decoding
succeeds asymptotically (x_l -> 0) iff epsilon is below the ensemble's
decoding threshold epsilon*. See Richardson & Urbanke, "Modern Coding
Theory", Theorem 3.50, or the original Luby et al. / Richardson-Urbanke BEC
analysis. For the standard (3,6) rate-1/2 ensemble the literature threshold
is epsilon* ~= 0.4294.
"""

from __future__ import annotations


def de_step(x: float, epsilon: float, dv: int, dc: int) -> float:
    """One density-evolution update x_l -> x_{l+1}."""
    return epsilon * (1.0 - (1.0 - x) ** (dc - 1)) ** (dv - 1)


def de_converges_to_zero(
    epsilon: float,
    dv: int,
    dc: int,
    max_iters: int = 2000,
    tol: float = 1e-12,
) -> bool:
    """Whether the DE recursion started at x_0=epsilon converges to 0."""
    if epsilon <= 0.0:
        return True
    x = epsilon
    for _ in range(max_iters):
        x_next = de_step(x, epsilon, dv, dc)
        if x_next < tol:
            return True
        # A fixed point above tol with no further progress means the
        # recursion has stalled above zero: decoding fails asymptotically.
        if abs(x_next - x) < 1e-15:
            return False
        x = x_next
    return x < tol


def find_threshold(
    dv: int,
    dc: int,
    lo: float = 0.0,
    hi: float = 1.0,
    tol: float = 1e-6,
    max_iters: int = 2000,
) -> float:
    """Bisection search for the ensemble's BEC decoding threshold epsilon*.

    epsilon* = sup{epsilon in [0,1] : DE recursion started at x_0=epsilon
    converges to 0}.
    """
    if not de_converges_to_zero(lo, dv, dc, max_iters):
        raise ValueError("DE does not converge even at lo bound; bad lo")
    if de_converges_to_zero(hi, dv, dc, max_iters):
        return hi

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if de_converges_to_zero(mid, dv, dc, max_iters):
            lo = mid
        else:
            hi = mid
    return lo


def de_trajectory(epsilon: float, dv: int, dc: int, n_iters: int) -> list[float]:
    """Return x_0, x_1, ..., x_{n_iters} for inspection/plotting."""
    xs = [epsilon]
    x = epsilon
    for _ in range(n_iters):
        x = de_step(x, epsilon, dv, dc)
        xs.append(x)
    return xs
