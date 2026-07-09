"""Closed-form LSH theory for random-hyperplane (SimHash) hashing.

All formulas here are derived from Goemans-Williamson / Charikar (2002)
and the Indyk-Motwani (1998) LSH framework, not fit to data. The rest of
this project measures how closely a *from-scratch* implementation tracks
these formulas.
"""
from __future__ import annotations

import math


def single_hash_collision_prob(theta: float) -> float:
    """Pr[sign(r.u) == sign(r.v)] for a uniformly random hyperplane normal r,
    where theta = angle(u, v) in [0, pi].

    This is the classical Goemans-Williamson / Charikar (2002) identity:
    a random hyperplane separates two unit vectors iff its normal falls in
    one of the two "wedges" between them, which together have angular
    measure 2*theta out of the full 2*pi (or, restricting to a great circle
    through u and v, measure theta out of pi). Hence
        Pr[disagree] = theta / pi,   Pr[agree] = 1 - theta / pi.
    """
    if not (0.0 <= theta <= math.pi + 1e-9):
        raise ValueError(f"theta must be in [0, pi], got {theta}")
    theta = min(max(theta, 0.0), math.pi)
    return 1.0 - theta / math.pi


def banded_collision_prob(theta: float, k: int) -> float:
    """Pr[all k independent hash bits agree] = p(theta)^k (AND of k hashes,
    i.e. a single "band" / table signature of length k)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return single_hash_collision_prob(theta) ** k


def or_of_bands_prob(theta: float, k: int, L: int) -> float:
    """Pr[at least one of L independent length-k bands collides].

    This is the standard LSH "S-curve":
        Pr[hit] = 1 - (1 - p(theta)^k)^L.
    """
    if L < 1:
        raise ValueError("L must be >= 1")
    p_band = banded_collision_prob(theta, k)
    return 1.0 - (1.0 - p_band) ** L


def threshold_similarity(k: int, L: int) -> float:
    """The single-hash collision probability s* at which the AND-OR S-curve
    crosses 0.5 recall: solve 1 - (1 - s^k)^L = 0.5 for s.

    s* = (1 - 0.5^(1/L))^(1/k)
    """
    return (1.0 - 0.5 ** (1.0 / L)) ** (1.0 / k)


def threshold_angle(k: int, L: int) -> float:
    """Angle theta* corresponding to threshold_similarity(k, L), via
    s = 1 - theta/pi  =>  theta = pi * (1 - s)."""
    s_star = threshold_similarity(k, L)
    return math.pi * (1.0 - s_star)


def rho_exponent(p1: float, p2: float) -> float:
    """Indyk-Motwani (1998) sublinearity exponent for LSH-based ANN:
    query cost / candidate-set size scales as n^rho with
        rho = ln(1/p1) / ln(1/p2),   0 < p2 < p1 < 1.

    p1 is the single-hash collision probability at the "near" radius,
    p2 at the "far" radius (p1 > p2 since closer points collide more often).
    """
    if not (0.0 < p2 < p1 < 1.0):
        raise ValueError(f"require 0 < p2 < p1 < 1, got p1={p1}, p2={p2}")
    return math.log(1.0 / p1) / math.log(1.0 / p2)
