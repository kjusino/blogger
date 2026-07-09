"""Mean-field theory of signal propagation in infinite-width random MLPs.

Implements the variance/correlation recursions of Poole et al. (2016) and
Schoenholz et al. (2017): for a fully-connected network with i.i.d. Gaussian
weights W ~ N(0, sigma_w2 / N) and biases b ~ N(0, sigma_b2), pre-activations
become i.i.d. Gaussian in the infinite-width limit, and their variance /
cross-input correlation propagate through deterministic 1-D and 2-D maps.

All Gaussian expectations are evaluated with Gauss-Hermite quadrature rather
than generic numerical integration, so a full (q*, chi_1) evaluation is a
handful of vectorized dot products -- cheap enough to call thousands of times
in a grid sweep.
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

_N_QUAD = 60
_NODES, _WEIGHTS = hermegauss(_N_QUAD)
_NORM = np.sqrt(2.0 * np.pi)


def gaussian_expectation(f):
    """E_{z ~ N(0,1)}[f(z)] via Gauss-Hermite quadrature."""
    return float(np.sum(_WEIGHTS * f(_NODES)) / _NORM)


def q_map(q, sigma_w2, sigma_b2, phi):
    """One step of the pre-activation variance recursion q_{l+1} = V(q_l)."""
    if q < 0:
        raise ValueError("q must be non-negative")
    return sigma_w2 * gaussian_expectation(lambda z: phi(np.sqrt(q) * z) ** 2) + sigma_b2


def fixed_point_q(sigma_w2, sigma_b2, phi, q0=1.0, n_iter=2000, tol=1e-14):
    """Iterate q_map to its fixed point q*. The map is a contraction for any
    bounded-derivative activation, so plain iteration (no root finder) is
    robust and, empirically, always converges from q0=1 for the activations
    used here."""
    q = q0
    for _ in range(n_iter):
        q_new = q_map(q, sigma_w2, sigma_b2, phi)
        if abs(q_new - q) < tol:
            return q_new
        q = q_new
    return q


def chi1(q_star, sigma_w2, phi_prime):
    """Slope of the correlation map at c=1: chi_1 = sigma_w2 * E[phi'(sqrt(q*) z)^2].

    chi_1 < 1  -> c=1 is the stable fixed point ("ordered" phase): nearby
                  inputs' representations converge as depth grows.
    chi_1 > 1  -> c=1 is unstable ("chaotic" phase): nearby inputs'
                  representations separate exponentially with depth.
    chi_1 = 1  -> critical line ("edge of chaos"): the leading-order decay
                  rate vanishes and the correlation length diverges.
    """
    return sigma_w2 * gaussian_expectation(lambda z: phi_prime(np.sqrt(q_star) * z) ** 2)


def correlation_length(chi1_value):
    """xi_c = 1 / |ln(chi_1)|, the number of layers over which a small
    perturbation to c=1 decays (ordered phase, chi_1 < 1) or grows
    (chaotic phase, chi_1 > 1) by a factor of e. Using the absolute value
    makes xi_c a positive quantity that diverges symmetrically (in
    log-chi_1) as chi_1 -> 1 from either side -- the standard definition of
    a correlation length at a continuous phase transition, here applied to
    the edge-of-chaos transition of Poole et al. (2016)."""
    if chi1_value <= 0:
        return np.inf
    log_chi1 = np.log(chi1_value)
    if abs(log_chi1) < 1e-12:
        return np.inf
    return 1.0 / abs(log_chi1)


def correlation_map(c, q_star, sigma_w2, sigma_b2, phi):
    """One step of the correlation recursion c_{l+1} = f(c_l), evaluated at
    the variance fixed point q*, via 2-D Gauss-Hermite quadrature over the
    independent Gaussian pair (z1, z2) used to construct a bivariate normal
    with correlation c.
    """
    c = float(np.clip(c, -1.0, 1.0))
    sqrt_q = np.sqrt(q_star)
    sqrt_term = np.sqrt(max(1.0 - c * c, 0.0))

    z1 = _NODES[:, None]
    z2 = _NODES[None, :]
    w = _WEIGHTS[:, None] * _WEIGHTS[None, :]

    u1 = sqrt_q * z1
    u2 = sqrt_q * (c * z1 + sqrt_term * z2)
    integral = np.sum(w * phi(u1) * phi(u2)) / (_NORM ** 2)
    return (sigma_w2 * integral + sigma_b2) / q_star


def fixed_point_c(q_star, sigma_w2, sigma_b2, phi, c0=1e-3, n_iter=400):
    """Iterate the correlation map from a low starting correlation to find
    its attracting fixed point. c=1 is always a fixed point of the map but
    is *unstable* whenever chi_1 > 1, so starting away from 1 finds whichever
    fixed point actually attracts nearby trajectories (1 in the ordered
    phase, a sub-1 value in the chaotic phase) without special-casing."""
    if q_star <= 0:
        return 1.0
    c = c0
    for _ in range(n_iter):
        c = correlation_map(c, q_star, sigma_w2, sigma_b2, phi)
    return c


def analyze_point(sigma_w2, sigma_b2, phi, phi_prime):
    """Full mean-field summary at one (sigma_w2, sigma_b2) point."""
    q_star = fixed_point_q(sigma_w2, sigma_b2, phi)
    chi1_val = chi1(q_star, sigma_w2, phi_prime)
    xi_c = correlation_length(chi1_val)
    c_star = fixed_point_c(q_star, sigma_w2, sigma_b2, phi)
    return {
        "sigma_w2": sigma_w2,
        "sigma_b2": sigma_b2,
        "q_star": q_star,
        "chi1": chi1_val,
        "xi_c": xi_c,
        "c_star": c_star,
        "phase": "chaotic" if chi1_val > 1.0 else "ordered",
    }


def critical_sigma_w2(sigma_b2, phi, phi_prime, lo=1.0 + 1e-6, hi=8.0):
    """Solve chi_1(sigma_w2, sigma_b2) = 1 for sigma_w2 via bisection -- the
    edge-of-chaos critical line sigma_w2*(sigma_b2)."""
    from scipy.optimize import brentq

    def f(sw2):
        q = fixed_point_q(sw2, sigma_b2, phi)
        return chi1(q, sw2, phi_prime) - 1.0

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        # already chaotic at the smallest sigma_w2 tried (large sigma_b2 case)
        return lo
    if f_hi < 0:
        return hi
    return brentq(f, lo, hi)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    t = np.tanh(x)
    return 1.0 - t * t
