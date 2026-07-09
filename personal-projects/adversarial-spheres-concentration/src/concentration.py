"""Exact spherical-cap measure and the Levy-concentration adversarial-robustness ceiling.

Core fact (Levy's isoperimetric theorem on S^{d-1}): among all measurable subsets
of the sphere with a given normalized measure, spherical caps minimize the measure
of their epsilon-blow-up's complement. This gives an exact, non-asymptotic *ceiling*
on how robust any binary classifier's decision boundary can be, as a function of
the minority-class decision-region measure and the ambient dimension.
"""

import numpy as np
from scipy.special import betainc, betaincinv
from scipy.optimize import brentq


def cap_measure(theta, d):
    """Normalized surface measure of a spherical cap of half-angle theta on S^{d-1}.

    theta in [0, pi]; d = ambient dimension (points live in R^d, sphere is S^{d-1}).
    """
    theta = np.asarray(theta, dtype=float)
    x = np.sin(theta) ** 2
    a = (d - 1) / 2
    b = 0.5
    half = betainc(a, b, x) / 2
    return np.where(theta <= np.pi / 2, half, 1 - half)


def cap_angle_for_measure(p, d):
    """Inverse of cap_measure: half-angle theta such that cap_measure(theta, d) == p."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return np.pi
    if p <= 0.5:
        x = betaincinv((d - 1) / 2, 0.5, 2 * p)
        return float(np.arcsin(np.sqrt(x)))
    else:
        return np.pi - cap_angle_for_measure(1 - p, d)


def chord_from_geodesic(radius, phi):
    """Euclidean chord length between two points on a sphere of given radius
    separated by geodesic angle phi."""
    return 2 * radius * np.sin(phi / 2)


def levy_ceiling_exact(p_minor, d, radius, delta=0.5):
    """Exact isoperimetric ceiling on adversarial robustness.

    Given a classifier whose decision regions on a sphere of the given radius have
    minority-class measure `p_minor` (<=0.5), Levy's isoperimetric theorem says the
    *worst-case* (i.e. most-robust-possible) shape for the majority region is a
    spherical cap. This function returns the Euclidean perturbation distance
    epsilon such that, even in that worst case, only a `delta` fraction of the
    minority region can be farther than epsilon from the majority region's boundary.
    I.e. no classifier with this class balance can have more than a `delta`
    fraction of points with robustness margin exceeding the returned epsilon.

    Returns +inf if p_minor <= 0 (degenerate / perfectly separated case has no
    finite guaranteed ceiling from this argument).
    """
    if p_minor <= 0:
        return np.inf
    p_minor = min(p_minor, 0.5)
    theta0 = cap_angle_for_measure(1 - p_minor, d)  # majority region modeled as extremal cap

    def leftover(phi):
        # measure of points still farther than blow-up angle phi from the majority cap
        return (1 - cap_measure(theta0 + phi, d)) - delta * p_minor

    phi_max = np.pi - theta0
    if phi_max <= 0:
        return 0.0
    if leftover(0.0) <= 0:
        return 0.0
    if leftover(phi_max) > 0:
        # even blowing up to the antipode doesn't reach the target; ceiling is the
        # farthest possible chord (diameter)
        phi_star = phi_max
    else:
        phi_star = brentq(leftover, 0.0, phi_max)
    return chord_from_geodesic(radius, phi_star)


def levy_bound_asymptotic(eps, d, radius):
    """Standard asymptotic Levy-lemma tail bound exp(-(d-2) g^2 / 4) where g is the
    geodesic angle corresponding to Euclidean distance eps on the given sphere.
    Provided for comparison against the exact cap-based bound above.
    """
    g = 2 * np.arcsin(np.clip(eps / (2 * radius), -1, 1))
    return np.exp(-(d - 2) * g**2 / 4)
