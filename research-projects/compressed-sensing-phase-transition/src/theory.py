"""Closed-form prediction for the L1-minimization compressed-sensing phase
transition, derived from the statistical-dimension / conic-geometry theory
of Amelunxen, Lotz, McCoy & Tropp, "Living on the Edge: A Geometric Theory
of Phase Transitions in Convex Optimization" (2014).

Setup
-----
x0 in R^n is exactly k-sparse. We observe y = A x0 for a random Gaussian
sensing matrix A in R^{m x n}, and recover x0 by solving the basis-pursuit
linear program

    minimize ||x||_1  subject to  A x = y.

Recovery succeeds with overwhelming probability (as n -> infinity) once the
number of measurements m exceeds a sharp threshold that depends only on the
normalized sparsity rho = k / n. ALMT's theory identifies that threshold as
the *statistical dimension* of the descent cone of the l1 norm at x0,
normalized by n:

    delta_c(rho) = psi(rho) := min_{lam >= 0} f(rho, lam)

where, writing Phi/phi for the standard normal CDF/PDF,

    f(rho, lam) = rho * (1 + lam^2)
                  + (1 - rho) * 2 * [ (1 + lam^2) * Phi(-lam) - lam * phi(lam) ]

Derivation sketch (also in README.md): the subdifferential of ||.||_1 at a
k-sparse x0 is the set of vectors that agree with sign(x0) on the support and
lie in [-1, 1] off the support. For g ~ N(0, I_n), the squared distance from
g to lam * (this set), summed over coordinates and normalized by n, has
expectation f(rho, lam): each of the k support coordinates contributes
E[(g_i - lam)^2] = 1 + lam^2, and each of the n-k off-support coordinates
contributes E[dist(g_i, [-lam, lam])^2] = 2*[(1+lam^2)*Phi(-lam) - lam*phi(lam)],
a standard truncated-Gaussian second moment. Minimizing the lam-dependent
upper bound over lam >= 0 gives the (asymptotically tight, per ALMT Cor. 3.3
+ Gordon's escape-through-a-mesh theorem) statistical dimension fraction.

We work throughout in delta = m/n, rho = k/n (both normalized by the ambient
dimension n) — NOT the delta = m/n, rho = k/m convention used in some of the
older Donoho-Tanner literature. Under Donoho-Tanner's convention the same
transition curve is asymptotically equivalent (their random-polytope
neighborliness result and the ALMT statistical-dimension result describe the
same limiting phase transition), but the two conventions are related by a
simple change of variables, so this module's rho must not be compared
numerically against a rho = k/m curve without converting first.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def statistical_dimension_fraction(rho: float, lam: float) -> float:
    """f(rho, lam): normalized expected squared distance from a standard
    Gaussian vector to lam times the l1-norm subdifferential at a
    rho-sparse vector (rho = k / n). See module docstring for the formula.

    Vectorized over `lam` (accepts a scalar or numpy array).
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must lie in [0, 1], got {rho}")
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 0):
        raise ValueError("lam must be non-negative")

    support_term = rho * (1.0 + lam**2)
    off_support_term = (1.0 - rho) * 2.0 * (
        (1.0 + lam**2) * norm.cdf(-lam) - lam * norm.pdf(lam)
    )
    return support_term + off_support_term


def phase_transition_delta(rho: float) -> float:
    """psi(rho): the ALMT-predicted critical delta = m/n above which
    l1-minimization recovers a rho = k/n sparse vector with probability
    tending to 1 as n -> infinity, and below which it fails with
    probability tending to 1.

    Returns a value in [0, 1] (statistical dimension of a cone in R^n can
    never exceed n).
    """
    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return 1.0

    result = minimize_scalar(
        lambda lam: statistical_dimension_fraction(rho, lam),
        bounds=(0.0, 50.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(np.clip(result.fun, 0.0, 1.0))


def phase_transition_curve(rho_values: np.ndarray) -> np.ndarray:
    """Vectorized convenience wrapper around `phase_transition_delta`."""
    return np.array([phase_transition_delta(rho) for rho in rho_values])
