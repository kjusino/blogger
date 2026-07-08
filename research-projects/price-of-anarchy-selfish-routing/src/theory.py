"""The topology-independent price-of-anarchy bound for polynomial latencies,
and an independent closed-form two-link ("Pigou-style") derivation used to
cross-check it numerically.

Roughgarden's bound (Roughgarden, "The Price of Anarchy Is Independent of
the Network Topology", JCSS 2003; see also Roughgarden & Tardos, "How Bad
is Selfish Routing?", JACM 2002 for the p=1 case): for nonatomic routing
games whose edge-latency functions are polynomials of degree at most `p`
with nonnegative coefficients, the price of anarchy (equilibrium cost /
optimal cost) is at most

    beta(p) = 1 / (1 - p * (p + 1)^(-(p+1)/p))

*regardless of the network topology*. For p = 1 this is the famous 4/3
bound for affine (linear) latencies.

`worst_case_two_link_poa` independently re-derives the same number from
first principles by solving, in closed form, the equilibrium and optimal
flow on the simplest possible network that can realize the bound: two
parallel links between a single source and sink, one with latency x^p and
one with a constant latency b, then numerically searching over b for the
ratio-maximizing instance. Roughgarden's topology-independence theorem
says this simplest possible network already achieves the worst case over
*all* topologies -- `tests/test_theory.py` checks that this independent,
from-scratch calculation agrees with the closed-form `beta(p)` formula,
and `experiment.py`/`equilibrium.py` separately re-derive the same number
a *third* way, via a generic nonlinear-program solver applied to that same
two-link network, so the bound is validated by three independent routes.
"""

from __future__ import annotations

from scipy.optimize import minimize_scalar


def poa_bound(p: int) -> float:
    """Roughgarden's topology-independent PoA bound for degree-<=p polynomial
    latencies with nonnegative coefficients."""
    if p < 1:
        raise ValueError("p must be >= 1")
    return 1.0 / (1.0 - p * (p + 1) ** (-(p + 1) / p))


def two_link_equilibrium_cost(p: int, b: float) -> float:
    """Closed-form equilibrium cost on the two-link network: edge 1 has
    latency x^p, edge 2 has constant latency b, unit demand.

    If b <= 1: both edges are used, each at common latency b (equilibrium
    condition), so total cost = b * (total flow) = b.
    If b > 1: edge 1 alone is cheaper at full load (cost 1 < b), so all
    flow uses edge 1 and the equilibrium cost is 1.
    """
    if b < 0:
        raise ValueError("b must be >= 0")
    return b if b <= 1.0 else 1.0


def two_link_optimum_cost(p: int, b: float) -> float:
    """Closed-form social-optimum cost for the same network.

    Minimize f1^(p+1) + b*(1 - f1) over f1 in [0, 1]. The unconstrained
    stationary point is f1* = (b / (p+1))^(1/p); clip to [0, 1] since C is
    convex in f1 (both f1^(p+1) and -b*f1 are... the objective is convex
    because f1^(p+1) is convex for p+1 >= 1 and the rest is linear).
    """
    if b < 0:
        raise ValueError("b must be >= 0")
    f1_star = (b / (p + 1)) ** (1.0 / p)
    f1_star = min(max(f1_star, 0.0), 1.0)
    f2_star = 1.0 - f1_star
    return f1_star ** (p + 1) + b * f2_star


def two_link_poa(p: int, b: float) -> float:
    """PoA(b) = equilibrium cost / optimal cost for the two-link network."""
    opt = two_link_optimum_cost(p, b)
    if opt <= 0:
        return 1.0
    return two_link_equilibrium_cost(p, b) / opt


def worst_case_two_link_poa(p: int) -> tuple:
    """Numerically maximize two_link_poa(p, b) over b in [0, p+1] (the range
    over which the optimum's stationary point can be interior). Returns
    (best_b, best_poa) -- this is the from-scratch numeric derivation
    cross-checked against `poa_bound(p)` in the tests."""
    neg = lambda b: -two_link_poa(p, b)
    result = minimize_scalar(neg, bounds=(1e-9, p + 1), method="bounded",
                              options={"xatol": 1e-12})
    best_b = result.x
    return best_b, two_link_poa(p, best_b)
